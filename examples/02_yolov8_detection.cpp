#include "axflow/axflow.h"
#include "common/args.h"
#include "common/yolov8_utils.h"
#include "common/viz.h"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace axflow;

int main(int argc, char** argv)
{
    auto args = common::parse_detection_args(argc, argv);
    if (!args) return 1;

    cv::Mat image = cv::imread(args->image_path);
    if (image.empty()) return 1;

    Device dev;
    AxFlow flow(dev, args->config_path);
    flow.preprocess(image);
    flow.inference();
    auto tensors = flow.postamble();
    auto detections = common::parse_yolov8_detections(
        tensors, image.cols, image.rows, 0.25f, 0.45f);

    common::draw_detections(image, detections);
    std::cout << "Detections: " << detections.size() << std::endl;
    for (const auto& d : detections)
        std::cout << "  " << common::COCO_CLASSES[d.class_id]
            << " " << d.score << "\n";

    cv::imwrite(args->output_path, image);
    return 0;
}
