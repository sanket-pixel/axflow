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
            throw std::runtime_error("yolov8 post-process received empty tensors");

        // COCO postamble output: [1, 84, 8400]
        // layout: [batch, 4_coords + 80_classes, num_anchors]
        const auto& det = tensors[0];
        const float* data = det.data.data();

        // shape[0]=1, shape[1]=84, shape[2]=8400
        const int num_anchors = det.shape[2]; // 8400
        const int num_classes = det.shape[1] - 4; // 80

        int best_idx = -1;
        float best_score = 0.0f;
        int best_class = -1;

        for (int i = 0; i < num_anchors; ++i)
        {
            // coords are at [0..3, i], classes at [4..83, i]
            // data layout: row-major [84, 8400]
            // so class j at anchor i = data[j * num_anchors + i]
            float max_cls_score = 0.0f;
            int max_cls = -1;
            for (int c = 0; c < num_classes; ++c)
            {
                float s = data[(4 + c) * num_anchors + i];
                if (s > max_cls_score)
                {
                    max_cls_score = s;
                    max_cls = c;
                }
            }

            if (max_cls_score > best_score)
            {
                best_score = max_cls_score;
                best_idx = i;
                best_class = max_cls;
            }
        }

        if (best_idx < 0)
        {
            std::cout << "warning: no detections found\n";
            return Detection();
        }

        // coords at rows 0-3, column best_idx
        float cx = data[0 * num_anchors + best_idx];
        float cy = data[1 * num_anchors + best_idx];
        float w = data[2 * num_anchors + best_idx];
        float h = data[3 * num_anchors + best_idx];

        // COCO model input is 640x640 square
        const float scale_x = static_cast<float>(orig_width) / 640.0f;
        const float scale_y = static_cast<float>(orig_height) / 640.0f;

        Detection result;
        result.x1 = (cx - w / 2.0f) * scale_x;
        result.y1 = (cy - h / 2.0f) * scale_y;
        result.x2 = (cx + w / 2.0f) * scale_x;
        result.y2 = (cy + h / 2.0f) * scale_y;
        result.score = best_score;
        result.class_id = best_class;

        return result;
    }
}
