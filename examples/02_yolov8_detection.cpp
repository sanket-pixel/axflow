#include "axflow/axflow.h"
#include "common/args.h"
#include "common/yolov8_utils.h"
#include "common/viz.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace axflow;

int main(int argc, char **argv) {
    auto args = common::parse_detection_args(argc, argv);
    if (!args) return 1;

    cv::Mat image = cv::imread(args->image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image at " << args->image_path << "\n";
        return 1;
    }

    Device dev;
    AxFlow flow(dev, args->config_path);

    flow.preprocess(image);
    const auto &tensors = flow.inference();
    auto detections = common::parse_yolov8_detections(
        tensors, image.cols, image.rows, 0.25f, 0.45f);
    std::cout << "detections: " << detections.size() << "\n";
    for (const auto &d: detections) {
        std::cout << "  " << common::COCO_CLASSES[d.class_id]
                << " " << d.score << "\n";
    }
    common::draw_detections(image, detections);
    cv::imwrite(args->output_path, image);

    return 0;
}
