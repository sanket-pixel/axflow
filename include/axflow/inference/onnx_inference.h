#pragma once

#include "axflow/config/onnx_inference_config.h"
#include "axflow/data_types/tensor.h"
#include "axflow/inference/inference_interface.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace axflow {
    // Runs an uncompiled ONNX heavy model graph purely on the CPU via ONNXRuntime.
    class OnnxInference : public InferenceInterface {
    public:
        explicit OnnxInference(const OnnxInferenceConfig &cfg);

        ~OnnxInference() override;

        OnnxInference(const OnnxInference &) = delete;

        OnnxInference &operator=(const OnnxInference &) = delete;

        OnnxInference(OnnxInference &&) noexcept = default;

        OnnxInference &operator=(OnnxInference &&) noexcept = default;

        int input_count() const override { return static_cast<int>(input_names_.size()); }
        int output_count() const override { return static_cast<int>(output_names_.size()); }

        const std::string &input_name(int index) const override;

        const std::string &output_name(int index) const override;

        std::vector<Tensor> &run() override;

        Tensor &get_input_tensor(int index = 0);

        Tensor &get_input_tensor(const std::string &name);

        const Tensor &get_output_tensor(const std::string &name) const;

        const Tensor &get_output_tensor(int index) const;

    private:
        int find_input_index(const std::string &name) const;

        int find_output_index(const std::string &name) const;

        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "axflow_ort_inference"};
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_{nullptr};
        std::unique_ptr<Ort::IoBinding> io_binding_;

        std::vector<Ort::AllocatedStringPtr> input_name_holders_;
        std::vector<Ort::AllocatedStringPtr> output_name_holders_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

        // The Engine's Physical Memory
        std::vector<Tensor> input_tensors_;
        std::vector<Tensor> output_tensors_;

        std::vector<std::vector<int64_t> > input_shapes_;
        std::vector<std::vector<int64_t> > output_shapes_;

        // Statically bound ORT tensor wrappers
        std::vector<Ort::Value> ort_inputs_;
        std::vector<Ort::Value> ort_outputs_;

        std::vector<int64_t> get_sanitized_shape(const Ort::TypeInfo &type_information) const;

        void initialize_inputs(Ort::AllocatorWithDefaultOptions &allocator);

        void initialize_outputs(Ort::AllocatorWithDefaultOptions &allocator);
    };
} // namespace axflow
