// BASE_FILE: examples/common/yolov8_utils.h
#pragma once
#include "axflow/data_types/tensor.h"
#include <vector>

namespace common
{
    struct Detection
    {
        float x1, y1, x2, y2;
        float score;
        int class_id = -1; // add this
    };

    static const std::vector<std::string> COCO_CLASSES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // Parses the [1, 27200, 5] postamble tensor, finds the highest score anchor,
    // and scales the coordinates back to the original image dimensions.
    std::vector<Detection> parse_yolov8_detections(
        const std::vector<axflow::Tensor>& tensors,
        int orig_width,
        int orig_height,
        float conf_threshold,
        float nms_threshold);
}
