#pragma once
#include "axflow/axflow.h"
#include "yolov8_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

namespace common
{
    struct SegDetection
    {
        float x1, y1, x2, y2;
        float score;
        int class_id;
        cv::Mat mask; // binary mask, same size as original image
    };

    // Parse YOLOv8-seg outputs into detections with masks
    // output0: [1, 116, 8400] — 4 box + 80 cls + 32 mask coefficients
    // output1: [1, 32, 160, 160] — prototype masks
    std::vector<SegDetection> parse_yolov8_seg_detections(
        const std::vector<axflow::Tensor>& tensors,
        int orig_width,
        int orig_height,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.45f);
} // namespace common
