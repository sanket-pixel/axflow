// BASE_FILE: examples/common/yolov8_utils.h
#pragma once
#include "axflow/data_types/tensor.h"
#include <vector>

namespace common
{
    struct Detection
    {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        float score = 0.0f;
    };

    // Parses the [1, 27200, 5] postamble tensor, finds the highest score anchor,
    // and scales the coordinates back to the original image dimensions.
    Detection parse_yolov8_best_box(const std::vector<axflow::Tensor>& tensors,
                                    int orig_width,
                                    int orig_height);
}
