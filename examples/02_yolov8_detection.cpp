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

    const int warmup_frames = 10;
    const int bench_frames = 100;

    // Warmup
    for (int i = 0; i < warmup_frames; ++i)
    {
        flow.preprocess(image);
        flow.inference();
        flow.postamble();
    }

    // Timing accumulators (microseconds)
    double total_pre = 0.0;
    double total_inf = 0.0;
    double total_post = 0.0;

    std::vector<axflow::Tensor> tensors;

    for (int i = 0; i < bench_frames; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        flow.preprocess(image);
        auto t1 = std::chrono::high_resolution_clock::now();
        flow.inference();
        auto t2 = std::chrono::high_resolution_clock::now();
        tensors = flow.postamble();
        auto t3 = std::chrono::high_resolution_clock::now();

        total_pre += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_inf += std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_post += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    double avg_pre = total_pre / bench_frames;
    double avg_inf = total_inf / bench_frames;
    double avg_post = total_post / bench_frames;
    double avg_total = avg_pre + avg_inf + avg_post;

    std::cout << "\n=== AxFlow Benchmark (" << bench_frames << " frames) ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Preprocess : " << avg_pre << " ms\n";
    std::cout << "  Inference  : " << avg_inf << " ms\n";
    std::cout << "  Postamble  : " << avg_post << " ms\n";
    std::cout << "  Total      : " << avg_total << " ms\n";
    std::cout << "  FPS        : " << (1000.0 / avg_total) << "\n";
    std::cout << "=====================================\n\n";

    // Final inference for visualization
    auto detections = common::parse_yolov8_detections(
        tensors, image.cols, image.rows, 0.25f, 0.45f);

    common::draw_detections(image, detections);
    std::cout << "Detections: " << detections.size() << "\n";
    for (const auto& d : detections)
        std::cout << "  " << common::COCO_CLASSES[d.class_id]
            << " " << d.score << "\n";

    cv::imwrite(args->output_path, image);
    return 0;
}
