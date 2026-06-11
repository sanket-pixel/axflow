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
    auto bbox = common::parse_yolov8_best_box(tensors, image.cols, image.rows);

    common::draw_detection(image, bbox);
    cv::imwrite(args->output_path, image);
    return 0;
}
