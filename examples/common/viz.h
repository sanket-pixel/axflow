// BASE_FILE: examples/common/viz.h
#pragma once
#include "yolov8_utils.h"
#include "yolov8_seg_utils.h"
#include <opencv2/core.hpp>

namespace common
{
    void draw_detection(cv::Mat& image, const Detection& det);
    void draw_detections(cv::Mat& image, const std::vector<Detection>& dets);
    // Add to existing viz.h
    void draw_seg_detections(cv::Mat& image,
                             const std::vector<SegDetection>& dets,
                             float mask_alpha = 0.4f);
}
