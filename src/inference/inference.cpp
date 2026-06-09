#include "axflow/inference/inference.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace axflow
{
  namespace
  {
    std::vector<int> shape_of(const axrTensorInfo& t)
    {
      return std::vector<int>(t.dims, t.dims + t.ndims);
    }

    std::vector<std::array<size_t, 2>> padding_of(const axrTensorInfo& t)
    {
      std::vector<std::array<size_t, 2>> p(t.ndims);
      for (int i = 0; i < t.ndims; ++i)
      {
        p[i] = {t.padding[i][0], t.padding[i][1]};
      }
      return p;
    }

    // dequantize int8 NHWC (padded) → float32 NCHW (dense)
    Tensor dequantize(const axrTensorInfo& info,
                      const std::vector<int8_t>& int8_buf,
                      const std::string& name)
    {
      // dequantize int8 NHWC (padded, chip layout) → float32 NCHW (dense).
      //
      // Assumes:
      //   - info.ndims == 4
      //   - layout is NHWC (the only layout supported by AxRuntime today)
      //   - padding may exist on any dim; we strip it
      //
      // For non-4D tensors (e.g. classifier heads with shape [N, num_classes])
      // this will throw. v2 will support arbitrary ranks and layouts.
      if (info.ndims != 4)
      {
        throw std::runtime_error(
          "axflow::dequantize: only 4D tensors supported, got " +
          std::to_string(info.ndims) + " dims for '" + name + "'");
      }
      // unpadded dimensions
      const int N = info.dims[0] - info.padding[0][0] - info.padding[0][1];
      const int H = info.dims[1] - info.padding[1][0] - info.padding[1][1];
      const int W = info.dims[2] - info.padding[2][0] - info.padding[2][1];
      const int C = info.dims[3] - info.padding[3][0] - info.padding[3][1];
      if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
      {
        throw std::runtime_error(
          "axflow::dequantize: invalid unpadded shape for '" + name + "'");
      }
      const int padded_H = info.dims[1];
      const int padded_W = info.dims[2];
      const int padded_C = info.dims[3];
      const int pad_h = info.padding[1][0];
      const int pad_w = info.padding[2][0];
      const int pad_c = info.padding[3][0];

      const float scale = info.scale;
      const int zp = info.zero_point;

      Tensor out;
      out.name = name;
      out.shape = {N, C, H, W};
      out.data.resize(static_cast<std::size_t>(N) * C * H * W);

      for (int n = 0; n < N; ++n)
      {
        for (int h = 0; h < H; ++h)
        {
          for (int w = 0; w < W; ++w)
          {
            for (int c = 0; c < C; ++c)
            {
              const int in_idx =
                ((n * padded_H + (h + pad_h)) * padded_W + (w + pad_w)) *
                padded_C +
                (c + pad_c);
              const int out_idx = ((n * C + c) * H + h) * W + w;
              out.data[out_idx] =
                (static_cast<float>(int8_buf[in_idx]) - zp) * scale;
            }
          }
        }
      }
      return out;
    }
  } // anonymous namespace

  Model::Model(Device& device, const AxflowConfig& cfg)
  {
    if (!device.is_connected())
    {
      throw std::runtime_error("axflow::Model: device not connected");
    }
    context_ = device.context();
    connection_ = device.connection();

    auto mpath = std::filesystem::path(cfg.inference.model_dir) / "model.json";

    model_ = axr_load_model(context_, mpath.string().c_str());
    if (!model_)
    {
      throw std::runtime_error(
        "axflow::Model: load_model failed: " +
        std::string(axr_last_error_string(AXR_OBJECT(context_))));
    }

    const std::string props_str = "input_dmabuf=0;num_sub_devices=1;aipu_cores=" +
      std::to_string(cfg.inference.num_cores);
    auto* props = axr_create_properties(context_, props_str.c_str());

    instance_ = axr_load_model_instance(connection_, model_, props);
    if (!instance_)
    {
      throw std::runtime_error(
        "axflow::Model: instance creation failed: " +
        std::string(axr_last_error_string(AXR_OBJECT(context_))));
    }

    const int n_in = axr_num_model_inputs(model_);
    const int n_out = axr_num_model_outputs(model_);

    input_infos_.resize(n_in);
    input_names_.resize(n_in);
    input_buffers_.resize(n_in);
    for (int i = 0; i < n_in; ++i)
    {
      input_infos_[i] = axr_get_model_input(model_, i);
      input_names_[i] = input_infos_[i].name ? input_infos_[i].name : "";
      input_buffers_[i].resize(axr_tensor_size(&input_infos_[i]));
    }

    output_infos_.resize(n_out);
    output_names_.resize(n_out);
    output_buffers_.resize(n_out);
    for (int i = 0; i < n_out; ++i)
    {
      output_infos_[i] = axr_get_model_output(model_, i);
      output_names_[i] = output_infos_[i].name ? output_infos_[i].name : "";
      output_buffers_[i].resize(axr_tensor_size(&output_infos_[i]));
    }
  }

  Model::~Model()
  {
    if (instance_)
      axr_destroy(reinterpret_cast<const axrObject*>(instance_));
    if (model_)
      axr_destroy(reinterpret_cast<const axrObject*>(model_));
  }

  int Model::find_input_index(const std::string& name) const
  {
    for (std::size_t i = 0; i < input_names_.size(); ++i)
    {
      if (input_names_[i] == name)
        return static_cast<int>(i);
    }
    throw std::runtime_error("axflow::Model: no input named '" + name + "'");
  }

  int Model::find_output_index(const std::string& name) const
  {
    for (std::size_t i = 0; i < output_names_.size(); ++i)
    {
      if (output_names_[i] == name)
        return static_cast<int>(i);
    }
    throw std::runtime_error("axflow::Model: no output named '" + name + "'");
  }

  InputBuffer Model::make_input_buffer(int i)
  {
    const auto& info = input_infos_.at(i);
    auto& buf = input_buffers_.at(i);
    return InputBuffer(input_names_.at(i), shape_of(info), padding_of(info),
                       info.scale, info.zero_point, buf.data(), buf.size());
  }

  InputBuffer Model::get_input(const std::string& name)
  {
    return make_input_buffer(find_input_index(name));
  }

  InputBuffer Model::get_input(int index) { return make_input_buffer(index); }

  std::vector<Tensor> Model::run()
  {
    // wire input/output arguments
    std::vector<axrArgument> input_args(input_buffers_.size());
    std::vector<axrArgument> output_args(output_buffers_.size());

    for (std::size_t i = 0; i < input_buffers_.size(); ++i)
    {
      input_args[i] = {input_buffers_[i].data(), 0, 0};
    }
    for (std::size_t i = 0; i < output_buffers_.size(); ++i)
    {
      output_args[i] = {output_buffers_[i].data(), 0, 0};
    }

    auto rc =
      axr_run_model_instance(instance_, input_args.data(), input_args.size(),
                             output_args.data(), output_args.size());
    if (rc != AXR_SUCCESS)
    {
      throw std::runtime_error(
        "axflow::Model: inference failed: " +
        std::string(axr_last_error_string(AXR_OBJECT(context_))));
    }

    // dequantize all outputs
    outputs_.clear();
    outputs_.reserve(output_infos_.size());
    for (std::size_t i = 0; i < output_infos_.size(); ++i)
    {
      outputs_.push_back(
        dequantize(output_infos_[i], output_buffers_[i], output_names_[i]));
    }
    return outputs_;
  }

  const Tensor& Model::get_output(const std::string& name) const
  {
    return outputs_.at(find_output_index(name));
  }

  const Tensor& Model::get_output(int index) const { return outputs_.at(index); }

  const std::string& Model::input_name(int i) const { return input_names_.at(i); }

  const std::string& Model::output_name(int i) const
  {
    return output_names_.at(i);
  }
} // namespace axflow
