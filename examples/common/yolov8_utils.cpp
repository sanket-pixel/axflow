// BASE_FILE: examples/common/yolov8_utils.cpp
#include "yolov8_utils.h"
#include <stdexcept>
#include <iostream>

namespace common
{
    Detection parse_yolov8_best_box(const std::vector<axflow::Tensor>& tensors,
                                    int orig_width,
                                    int orig_height)
    {
        if (tensors.empty())
        {
            throw std::runtime_error("yolov8 post-process received empty tensors");
        }

        const auto& det = tensors[0];
        const float* data = det.data.data();
        const int num_anchors = det.shape[1]; // 27200

        int best_idx = -1;
        float best_score = 0.0f;

        for (int i = 0; i < num_anchors; ++i)
        {
            float score = data[i * 5 + 4];
            if (score > best_score)
            {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_idx < 0)
        {
            std::cout << "warning: no detections found above 0.0 score\n";
            return Detection();
        }

        const float* row = data + best_idx * 5;
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        // scale model input space (640x512) back to original application space
        const float scale_x = static_cast<float>(orig_width) / 640.0f;
        const float scale_y = static_cast<float>(orig_height) / 512.0f;

        Detection result;
        result.x1 = (cx - w / 2.0f) * scale_x;
        result.y1 = (cy - h / 2.0f) * scale_y;
        result.x2 = (cx + w / 2.0f) * scale_x;
        result.y2 = (cy + h / 2.0f) * scale_y;
        result.score = best_score;

        return result;
    }
}
