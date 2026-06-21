#pragma once

#include "axflow/config/axflow_config.h"
#include "axflow/data_types/tensor.h"
#include "axflow/device/device.h"
#include "axflow/inference/inference_interface.h"
#include "axflow/postamble/postamble.h"
#include "axflow/preprocess/preprocessor.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace axflow {
    class AxFlow {
    public:
        AxFlow(Device &device, const std::string &config_path);
        AxFlow(Device &device, const AxflowConfig &config);

        AxFlow(const AxFlow &) = delete;
        AxFlow &operator=(const AxFlow &) = delete;
        AxFlow(AxFlow &&) noexcept = default;
        AxFlow &operator=(AxFlow &&) noexcept = default;

        // ─── Pipeline Stages ───

        void preprocess(const cv::Mat &image);

        // Returns a reference to the engine's internal memory (Zero allocations)
        std::vector<Tensor> &inference();

        std::vector<Tensor> &postamble();

        // ─── Introspection ───

        Preprocessor &preprocessor() { return preprocessor_; }
        bool postamble_enabled() const { return postamble_ != nullptr; }
        bool is_onnx_backend() const { return config_.inference_type == "onnx"; }

    private:
        void init_engines(Device &device);

        AxflowConfig config_;
        Preprocessor preprocessor_;

        // Polymorphic engine (hides ONNX vs AIPU completely)
        std::unique_ptr<InferenceInterface> engine_;
        std::unique_ptr<Postamble> postamble_;

        // Stores postamble outputs to safely return by reference
        std::vector<Tensor> postamble_outputs_;
    };
} // namespace axflow