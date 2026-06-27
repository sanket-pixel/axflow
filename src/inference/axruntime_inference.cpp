#include "axflow/inference/axruntime_inference.h"
#include "axflow/utils/dequantize.h"
#include "axflow/utils/tensor_layout.h"
#include "axflow/utils/quantize.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace axflow {
    AxRuntimeInference::AxRuntimeInference(Device &device, const AxruntimeInferenceConfig &cfg) {
        if (!device.is_connected()) {
            throw std::runtime_error("axflow::AxRuntimeInference: device not connected");
        }
        context_ = device.context();
        connection_ = device.connection();

        auto model_path = std::filesystem::path(cfg.model_dir) / "model.json";
        model_ = axr_load_model(context_, model_path.string().c_str());
        if (!model_) {
            throw std::runtime_error("axflow::AxRuntimeInference: load_model failed");
        }

        const std::string property_string = "input_dmabuf=0;num_sub_devices=4;aipu_cores=" + std::to_string(
                                                cfg.num_cores);
        auto *properties = axr_create_properties(context_, property_string.c_str());

        instance_ = axr_load_model_instance(connection_, model_, properties);
        if (!instance_) {
            throw std::runtime_error("axflow::AxRuntimeInference: instance creation failed");
        }

        const int input_count = axr_num_model_inputs(model_);
        const int output_count = axr_num_model_outputs(model_);

        // 1. Setup Raw Memory and Info
        input_infos_.resize(input_count);
        input_names_.resize(input_count);
        raw_input_memory_.resize(input_count);
        for (int i = 0; i < input_count; ++i) {
            input_infos_[i] = axr_get_model_input(model_, i);
            input_names_[i] = input_infos_[i].name ? input_infos_[i].name : "";
            raw_input_memory_[i].resize(axr_tensor_size(&input_infos_[i]));
        }

        output_infos_.resize(output_count);
        output_names_.resize(output_count);
        raw_output_memory_.resize(output_count);
        for (int i = 0; i < output_count; ++i) {
            output_infos_[i] = axr_get_model_output(model_, i);
            output_names_[i] = output_infos_[i].name ? output_infos_[i].name : "";
            raw_output_memory_[i].resize(axr_tensor_size(&output_infos_[i]));
        }

        // 2. Build Smart Wrappers and Driver Structure
        input_buffers_.reserve(input_count);
        input_args_.resize(input_count);
        for (int i = 0; i < input_count; ++i) {
            input_buffers_.push_back(make_input_buffer(i));
            input_args_[i] = {raw_input_memory_[i].data(), 0, 0};
        }

        output_args_.resize(output_count);
        for (int i = 0; i < output_count; ++i) {
            output_args_[i] = {raw_output_memory_[i].data(), 0, 0};
        }

        // 3. Pre-allocate Engine Float32 Memory
        outputs_.resize(output_count);
        internal_inputs_.resize(input_count);

        for (int i = 0; i < input_count; ++i) {
            const auto &buf = input_buffers_[i];
            internal_inputs_[i].name = buf.name();
            // Force C=3 (valid RGB layout) for the Preprocessor
            internal_inputs_[i].shape = {buf.shape()[0], buf.shape()[1], buf.shape()[2], 3};
            internal_inputs_[i].data.resize(internal_inputs_[i].numel(), 0.0f);
        }
    }

    AxRuntimeInference::~AxRuntimeInference() {
        if (instance_) axr_destroy(reinterpret_cast<const axrObject *>(instance_));
        if (model_) axr_destroy(reinterpret_cast<const axrObject *>(model_));
    }

    int AxRuntimeInference::find_input_index(const std::string &name) const {
        for (std::size_t i = 0; i < input_names_.size(); ++i) {
            if (input_names_[i] == name) return static_cast<int>(i);
        }
        throw std::runtime_error("axflow::AxRuntimeInference: no input named '" + name + "'");
    }

    int AxRuntimeInference::find_output_index(const std::string &name) const {
        for (std::size_t i = 0; i < output_names_.size(); ++i) {
            if (output_names_[i] == name) return static_cast<int>(i);
        }
        throw std::runtime_error("axflow::AxRuntimeInference: no output named '" + name + "'");
    }

    InputBuffer AxRuntimeInference::make_input_buffer(int i) {
        const auto &info = input_infos_.at(i);
        auto &buffer = raw_input_memory_.at(i);
        return InputBuffer(input_names_.at(i), shape_of(info), padding_of(info),
                           info.scale, info.zero_point, buffer.data(), buffer.size());
    }

    InputBuffer AxRuntimeInference::get_input_buffer(const std::string &name) {
        return input_buffers_.at(find_input_index(name));
    }

    InputBuffer AxRuntimeInference::get_input_buffer(int index) {
        return input_buffers_.at(index);
    }

    void AxRuntimeInference::quantize_valid_channels(const float *source_pixel, int8_t *destination_pixel,
                                                     int valid_channels, float scale, int zero_point) const {
        for (int channel = 0; channel < valid_channels; ++channel) {
            destination_pixel[channel] = axflow::quantize(source_pixel[channel], scale, zero_point);
        }
    }

    void AxRuntimeInference::zero_pad_channels(int8_t *destination_pixel,
                                               int valid_channels, int padded_channels) const {
        for (int channel = valid_channels; channel < padded_channels; ++channel) {
            destination_pixel[channel] = 0;
        }
    }

    void AxRuntimeInference::quantize_and_pack_nhwc(const Tensor &source, InputBuffer &destination) const {
        const float *source_data = source.data.data();
        int8_t *destination_data = destination.data();

        const float scale = destination.scale();
        const int zero_point = destination.zero_point();

        const int num_batches = source.shape[0];
        const int height = source.shape[1];
        const int width = source.shape[2];
        const int valid_channels = source.shape[3];

        const int padded_channels = destination.shape()[3];
        const int num_pixels = num_batches * height * width;

        for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
            const float *source_pixel = &source_data[pixel_index * valid_channels];
            int8_t *destination_pixel = &destination_data[pixel_index * padded_channels];

            quantize_valid_channels(source_pixel, destination_pixel, valid_channels, scale, zero_point);
            zero_pad_channels(destination_pixel, valid_channels, padded_channels);
        }
    }

    // ─── Implementation of the Interface ───

    Tensor &AxRuntimeInference::get_input_tensor(int index) {
        return internal_inputs_.at(index);
    }

    std::vector<Tensor> &AxRuntimeInference::run() {
        // 1. Pack and quantize all inputs
        for (std::size_t i = 0; i < internal_inputs_.size(); ++i) {
            quantize_and_pack_nhwc(internal_inputs_[i], input_buffers_[i]);
        }

        // 2. Execute hardware inference
        auto result = axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                                             output_args_.data(), output_args_.size());
        if (result != AXR_SUCCESS) {
            throw std::runtime_error("axflow::AxRuntimeInference: inference failed");
        }

        // 3. Dequantize directly into pre-allocated outputs
        for (std::size_t i = 0; i < output_infos_.size(); ++i) {
            outputs_[i] = axflow::dequantize_nhwc_to_nchw(
                output_infos_[i], raw_output_memory_[i], output_names_[i]);
        }

        return outputs_;
    }

    const Tensor &AxRuntimeInference::get_output_tensor(const std::string &name) const {
        return outputs_.at(find_output_index(name));
    }

    const Tensor &AxRuntimeInference::get_output_tensor(int index) const {
        return outputs_.at(index);
    }

    const std::string &AxRuntimeInference::input_name(int index) const {
        return input_names_.at(index);
    }

    const std::string &AxRuntimeInference::output_name(int index) const {
        return output_names_.at(index);
    }
} // namespace axflow
