// BASE_FILE: examples/common/viz.h
#pragma once
#include "yolov8_utils.h"
#include <opencv2/core.hpp>

namespace common
{
    void draw_detection(cv::Mat& image, const Detection& det);
    void draw_detections(cv::Mat& image, const std::vector<Detection>& dets);
}
