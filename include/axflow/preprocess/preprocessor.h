#pragma once

#include "axflow/config/preprocessor_config.h"
#include "axflow/data_types/input_buffer.h"
#include "axflow/data_types/tensor.h"
#include <opencv2/core.hpp>

namespace axflow {
    // Handles resizing, color conversion, normalization, and layout packing.
    class Preprocessor {
    public:
        explicit Preprocessor(const PreprocessingConfig &cfg);
        
        // Unified Scratchpad Path: Writes into a standard float32 NCHW Tensor.
        void run(const cv::Mat &image, Tensor &out_tensor) const;

    private:
        PreprocessingConfig cfg_;
    };
} // namespace axflow
