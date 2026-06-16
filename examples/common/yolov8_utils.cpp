// BASE_FILE: examples/common/yolov8_utils.cpp
#include "yolov8_utils.h"
#include <stdexcept>
#include <iostream>
#include <opencv2/dnn.hpp>

namespace common
{
    std::vector<Detection> parse_yolov8_detections(
        const std::vector<axflow::Tensor>& tensors,
        int orig_width,
        int orig_height,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.45f)
    {
        if (tensors.empty())
            throw std::runtime_error("empty tensors");

        const auto& det = tensors[0];
        const float* data = det.data.data();
        const int num_anchors = det.shape[2]; // 8400
        const int num_classes = det.shape[1] - 4; // 80

        const float scale_x = static_cast<float>(orig_width) / 640.0f;
        const float scale_y = static_cast<float>(orig_height) / 640.0f;

        // Step 1 — collect all detections above threshold
        std::vector<Detection> raw;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < num_anchors; ++i)
        {
            // find best class for this anchor
            float best_cls_score = 0.0f;
            int best_cls = -1;
            for (int c = 0; c < num_classes; ++c)
            {
                float s = data[(4 + c) * num_anchors + i];
                if (s > best_cls_score)
                {
                    best_cls_score = s;
                    best_cls = c;
                }
            }

            if (best_cls_score < conf_threshold) continue;

            float cx = data[0 * num_anchors + i];
            float cy = data[1 * num_anchors + i];
            float w = data[2 * num_anchors + i];
            float h = data[3 * num_anchors + i];

            float x1 = (cx - w / 2.0f) * scale_x;
            float y1 = (cy - h / 2.0f) * scale_y;
            float x2 = (cx + w / 2.0f) * scale_x;
            float y2 = (cy + h / 2.0f) * scale_y;

            boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            scores.push_back(best_cls_score);
            class_ids.push_back(best_cls);
        }

        // Step 2 — NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

        // Step 3 — collect final detections
        std::vector<Detection> results;
        for (int idx : indices)
        {
            Detection d;
            d.x1 = boxes[idx].x;
            d.y1 = boxes[idx].y;
            d.x2 = boxes[idx].x + boxes[idx].width;
            d.y2 = boxes[idx].y + boxes[idx].height;
            d.score = scores[idx];
            d.class_id = class_ids[idx];
            results.push_back(d);
        }

        return results;
    }
}
