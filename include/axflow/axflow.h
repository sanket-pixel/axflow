#pragma once

#include "axflow/config/axflow_config.h"
#include "axflow/data_types/input_buffer.h"
#include "axflow/data_types/tensor.h"
#include "axflow/device/device.h"
#include "axflow/inference/inference.h"
#include "axflow/postprocess/postprocessor.h"
#include "axflow/preprocess/preprocessor.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace axflow
{
    // top-level pipeline orchestrator.
    //
    // owns one Preprocessor, one Inference, one Postprocessor.
    // users drive the three stages explicitly:
    //
    //   Device dev;
    //   AxFlow flow(dev, "yolo.yaml");
    //
    //   flow.preprocess(image);
    //   flow.inference();
    //   auto results = flow.postprocess();
    //
    // for advanced use, components are accessible via accessors below.
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
        // returns dequantized float NCHW tensors (also cached internally).
        std::vector<Tensor> inference();

        // ── stage 3: postamble (currently stub pass-through) ──
        std::vector<Tensor> postprocess();

        // ── component access for advanced use ──
        Preprocessor& preprocessor() { return preprocessor_; }
        Inference& inference_obj() { return inference_; }
        Postprocessor& postprocessor() { return postprocessor_; }

    private:
        AxflowConfig config_;

        Preprocessor preprocessor_;
        Inference inference_;
        Postprocessor postprocessor_;

        // cached between stages
        std::vector<Tensor> inference_outputs_;
    };
} // namespace axflow
