#pragma once

#include "axflow/config/axflow_config.h"
#include "axflow/data_types/tensor.h"
#include "axflow/device/device.h"
#include "axflow/inference/inference.h"
#include "axflow/postamble/postamble.h"
#include "axflow/preprocess/preprocessor.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace axflow
{
    // top-level pipeline orchestrator.
    //
    // owns one Preprocessor, one Inference, and (if enabled) one Postamble.
    // users drive the stages explicitly:
    //
    //   Device dev;
    //   AxFlow flow(dev, "yolo.yaml");
    //
    //   flow.preprocess(image);
    //   flow.inference();
    //   auto tensors = flow.postamble();      // optional, throws if not enabled
    //
    //   // user-side postprocessing lives outside AxFlow.
    //
    class AxFlow
    {
    public:
        AxFlow(Device& device, const std::string& config_path);
        AxFlow(Device& device, const AxflowConfig& config);

        AxFlow(const AxFlow&) = delete;
        AxFlow& operator=(const AxFlow&) = delete;
        AxFlow(AxFlow&&) noexcept = default;
        AxFlow& operator=(AxFlow&&) noexcept = default;

        // ── stage 1: image → chip-side input buffer ──
        void preprocess(const cv::Mat& image);

        // ── stage 2: run on chip + dequantize ──
        std::vector<Tensor> inference();

        // ── stage 3: run ORT postamble graph ──
        // throws if postamble is not enabled in config.
        std::vector<Tensor> postamble();

        // ── component access for advanced use ──
        Preprocessor& preprocessor() { return preprocessor_; }
        Inference& inference_obj() { return inference_; }
        Postamble& postamble_obj(); // throws if not enabled

        bool postamble_enabled() const { return postamble_ != nullptr; }

    private:
        AxflowConfig config_;

        Preprocessor preprocessor_;
        Inference inference_;
        std::unique_ptr<Postamble> postamble_; // null if disabled in config

        // cached between stages
        std::vector<Tensor> inference_outputs_;
    };
} // namespace axflow
