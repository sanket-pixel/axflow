#pragma once

#include "axflow/config/inference_config.h"
#include "axflow/data_types/input_buffer.h"
#include "axflow/data_types/tensor.h"
#include "axflow/device/device.h"
#include "axruntime/axruntime.hpp"
#include "axflow/inference/inference_interface.h"

#include <cstdint>
#include <string>
#include <vector>

namespace axflow {
    class AxRuntimeInference : public InferenceInterface {
    public:
        AxRuntimeInference(Device &device, const InferenceConfig &cfg);

        ~AxRuntimeInference() override;

        AxRuntimeInference(const AxRuntimeInference &) = delete;

        AxRuntimeInference &operator=(const AxRuntimeInference &) = delete;

        AxRuntimeInference(AxRuntimeInference &&) noexcept = default;

        AxRuntimeInference &operator=(AxRuntimeInference &&) noexcept = default;

        // ─── InferenceInterface Contract ───

        int input_count() const override { return static_cast<int>(input_infos_.size()); }
        int output_count() const override { return static_cast<int>(output_infos_.size()); }

        const std::string &input_name(int index) const override;

        const std::string &output_name(int index) const override;

        Tensor &get_input_tensor(int index) override;

        std::vector<Tensor> &run() override;

        // ─── Native Hardware Layer 1 Primitives ───

        InputBuffer get_input_buffer(const std::string &name);

        InputBuffer get_input_buffer(int index = 0);

        const Tensor &get_output_tensor(const std::string &name) const;

        const Tensor &get_output_tensor(int index) const;

    private:
        axrContext *context_ = nullptr;
        axrConnection *connection_ = nullptr;
        axrModel *model_ = nullptr;
        axrModelInstance *instance_ = nullptr;

        std::vector<axrTensorInfo> input_infos_;
        std::vector<axrTensorInfo> output_infos_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

        // 1. The raw physical memory blocks mapping to the AIPU
        std::vector<std::vector<int8_t> > raw_input_memory_;
        std::vector<std::vector<int8_t> > raw_output_memory_;

        // 2. The smart wrappers for the raw memory
        std::vector<InputBuffer> input_buffers_;

        // 3. The driver structs for the AIPU
        std::vector<axrArgument> input_args_;
        std::vector<axrArgument> output_args_;

        // 4. The Engine's logical Float32 Memory
        std::vector<Tensor> internal_inputs_;
        std::vector<Tensor> outputs_;

        // ─── Helpers ───

        int find_input_index(const std::string &name) const;

        int find_output_index(const std::string &name) const;

        InputBuffer make_input_buffer(int index);

        void quantize_and_pack_nhwc(const Tensor &source, InputBuffer &destination) const;

        void quantize_valid_channels(const float *source_pixel, int8_t *destination_pixel,
                                     int valid_channels, float scale, int zero_point) const;

        void zero_pad_channels(int8_t *destination_pixel,
                               int valid_channels, int padded_channels) const;
    };
} // namespace axflow
