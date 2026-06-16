#include "yolov8_seg_utils.h"
#include <stdexcept>
#include <iostream>

namespace common
{
    std::vector<SegDetection> parse_yolov8_seg_detections(
        const std::vector<axflow::Tensor>& tensors,
        int orig_width,
        int orig_height,
        float conf_threshold,
        float nms_threshold)
    {
        if (tensors.size() < 2)
            throw std::runtime_error("yolov8-seg expects 2 output tensors, got "
                + std::to_string(tensors.size()));

        // output0: [1, 116, 8400]
        // 116 = 4 (box) + 80 (classes) + 32 (mask coefficients)
        const auto& det_tensor = tensors[0];
        const float* det_data = det_tensor.data.data();
        const int num_anchors = det_tensor.shape[2]; // 8400
        const int num_classes = 80;
        const int num_mask_coef = 32;

        // output1: [1, 32, 160, 160] — prototype masks
        const auto& proto_tensor = tensors[1];
        const float* proto_data = proto_tensor.data.data();
        const int proto_h = proto_tensor.shape[2]; // 160
        const int proto_w = proto_tensor.shape[3]; // 160

        const float scale_x = static_cast<float>(orig_width) / 640.0f;
        const float scale_y = static_cast<float>(orig_height) / 640.0f;

        // Step 1 — collect candidates above threshold
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<std::vector<float>> mask_coefficients;

        for (int i = 0; i < num_anchors; ++i)
        {
            // find best class
            float best_cls = 0.0f;
            int best_c = -1;
            for (int c = 0; c < num_classes; ++c)
            {
                float s = det_data[(4 + c) * num_anchors + i];
                if (s > best_cls)
                {
                    best_cls = s;
                    best_c = c;
                }
            }

            if (best_cls < conf_threshold) continue;

            float cx = det_data[0 * num_anchors + i];
            float cy = det_data[1 * num_anchors + i];
            float w = det_data[2 * num_anchors + i];
            float h = det_data[3 * num_anchors + i];

            float x1 = (cx - w / 2.0f) * scale_x;
            float y1 = (cy - h / 2.0f) * scale_y;
            float x2 = (cx + w / 2.0f) * scale_x;
            float y2 = (cy + h / 2.0f) * scale_y;

            // clamp to image bounds
            x1 = std::max(0.0f, std::min(x1, (float)orig_width));
            y1 = std::max(0.0f, std::min(y1, (float)orig_height));
            x2 = std::max(0.0f, std::min(x2, (float)orig_width));
            y2 = std::max(0.0f, std::min(y2, (float)orig_height));

            boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            scores.push_back(best_cls);
            class_ids.push_back(best_c);

            // collect 32 mask coefficients for this anchor
            std::vector<float> coefs(num_mask_coef);
            for (int m = 0; m < num_mask_coef; ++m)
                coefs[m] = det_data[(4 + num_classes + m) * num_anchors + i];
            mask_coefficients.push_back(coefs);
        }

        // Step 2 — NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

        // Step 3 — generate masks for surviving detections
        std::vector<SegDetection> results;
        results.reserve(indices.size());

        for (int idx : indices)
        {
            SegDetection det;
            det.x1 = boxes[idx].x;
            det.y1 = boxes[idx].y;
            det.x2 = boxes[idx].x + boxes[idx].width;
            det.y2 = boxes[idx].y + boxes[idx].height;
            det.score = scores[idx];
            det.class_id = class_ids[idx];

            // Generate mask: linear combination of prototypes
            // mask = sigmoid(sum(coef[m] * proto[m, :, :]))
            // proto shape: [32, 160, 160]
            cv::Mat mask_proto(proto_h, proto_w, CV_32F, 0.0f);
            const auto& coefs = mask_coefficients[idx];

            for (int m = 0; m < num_mask_coef; ++m)
            {
                // proto_data[m, h, w] = proto_data[m * proto_h * proto_w + h * proto_w + w]
                const float* proto_channel = proto_data + m * proto_h * proto_w;
                for (int ph = 0; ph < proto_h; ++ph)
                    for (int pw = 0; pw < proto_w; ++pw)
                        mask_proto.at<float>(ph, pw) +=
                            coefs[m] * proto_channel[ph * proto_w + pw];
            }

            // sigmoid
            cv::exp(-mask_proto, mask_proto);
            mask_proto = 1.0f / (1.0f + mask_proto);

            // resize proto mask from 160x160 to original image size
            cv::Mat mask_full;
            cv::resize(mask_proto, mask_full, cv::Size(orig_width, orig_height),
                       0, 0, cv::INTER_LINEAR);

            // threshold to binary
            cv::Mat mask_binary;
            cv::threshold(mask_full, mask_binary, 0.5f, 1.0f, cv::THRESH_BINARY);
            mask_binary.convertTo(det.mask, CV_8U, 255.0);

            // crop mask to bounding box only
            cv::Mat roi_mask = cv::Mat::zeros(orig_height, orig_width, CV_8U);
            cv::Rect bbox(static_cast<int>(det.x1), static_cast<int>(det.y1),
                          static_cast<int>(det.x2 - det.x1),
                          static_cast<int>(det.y2 - det.y1));
            bbox &= cv::Rect(0, 0, orig_width, orig_height); // clamp
            if (bbox.area() > 0)
                det.mask(bbox).copyTo(roi_mask(bbox));
            det.mask = roi_mask;

            results.push_back(det);
        }

        return results;
    }
} // namespace common
