#include "axflow/inference/inference.h"
#include "axflow/utils/dequantize.h"
#include "axflow/utils/tensor_layout.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace axflow
{
  Inference::Inference(Device& device, const AxflowConfig& cfg)
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

  Inference::~Inference()
  {
    if (instance_)
      axr_destroy(reinterpret_cast<const axrObject*>(instance_));
    if (model_)
      axr_destroy(reinterpret_cast<const axrObject*>(model_));
  }

  int Inference::find_input_index(const std::string& name) const
  {
    for (std::size_t i = 0; i < input_names_.size(); ++i)
    {
      if (input_names_[i] == name)
        return static_cast<int>(i);
    }
    throw std::runtime_error("axflow::Model: no input named '" + name + "'");
  }

  int Inference::find_output_index(const std::string& name) const
  {
    for (std::size_t i = 0; i < output_names_.size(); ++i)
    {
      if (output_names_[i] == name)
        return static_cast<int>(i);
    }
    throw std::runtime_error("axflow::Model: no output named '" + name + "'");
  }

  InputBuffer Inference::make_input_buffer(int i)
  {
    const auto& info = input_infos_.at(i);
    auto& buf = input_buffers_.at(i);
    return InputBuffer(input_names_.at(i), shape_of(info), padding_of(info),
                       info.scale, info.zero_point, buf.data(), buf.size());
  }

  InputBuffer Inference::get_input(const std::string& name)
  {
    return make_input_buffer(find_input_index(name));
  }

  InputBuffer Inference::get_input(int index) { return make_input_buffer(index); }

  std::vector<Tensor> Inference::run()
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
      outputs_.push_back(dequantize_nhwc_to_nchw(
        output_infos_[i], output_buffers_[i], output_names_[i]));
    }
    return outputs_;
  }

  const Tensor& Inference::get_output(const std::string& name) const
  {
    return outputs_.at(find_output_index(name));
  }

  const Tensor& Inference::get_output(int index) const { return outputs_.at(index); }

  const std::string& Inference::input_name(int i) const { return input_names_.at(i); }

  const std::string& Inference::output_name(int i) const
  {
    return output_names_.at(i);
  }
} // namespace axflow
