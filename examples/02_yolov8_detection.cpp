// examples/02_yolov8_detection.cpp
//
// run yolov8 ir detector via axflow end-to-end.
//   1. load image
//   2. axflow: preprocess → inference → postamble
//   3. find max-score anchor
//   4. draw box, save to disk
//
// usage:
//   02_yolov8_detection <config.yaml> <image.png> <output.png>

#include "axflow/axflow.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

using namespace axflow;

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0]
            << " <config.yaml> <image.png> <output.png>\n";
        return 1;
    }
    const std::string config_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = argv[3];

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "failed to load image: " << image_path << "\n";
        return 1;
    }
    std::cout << "image loaded: " << image.cols << "x" << image.rows << "\n";

    Device dev;
    AxFlow flow(dev, config_path);

    flow.preprocess(image);
    flow.inference();
    auto tensors = flow.postamble();

    if (tensors.empty())
    {
        std::cerr << "postamble produced no outputs\n";
        return 1;
    }

    // expected shape from ORT: [1, 27200, 5] — (cx, cy, w, h, score) per anchor
    const Tensor& det = tensors[0];
    std::cout << "postamble output '" << det.name << "' shape=[";
    for (int d : det.shape) std::cout << d << ",";
    std::cout << "]\n";

    if (det.shape.size() != 3 || det.shape[2] != 5)
    {
        std::cerr << "unexpected postamble shape, expected [1, N, 5]\n";
        return 1;
    }
    const int num_anchors = det.shape[1];

    // ── find max-score anchor ──
    const float* data = det.data.data();
    int best_idx = -1;
    float best_score = 0.0f;
    for (int i = 0; i < num_anchors; ++i)
    {
        const float score = data[i * 5 + 4];
        if (score > best_score)
        {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0)
    {
        std::cerr << "no detections above zero\n";
        return 1;
    }

    const float* row = data + best_idx * 5;
    const float cx = row[0], cy = row[1], w = row[2], h = row[3];
    std::cout << "best anchor " << best_idx
        << ": cx=" << cx << " cy=" << cy
        << " w=" << w << " h=" << h
        << " score=" << best_score << "\n";

    // ── coordinate space note ──
    // the postamble returns anchors in the *model input* coordinate space
    // (640 wide × 512 tall for yolov8s-ir). scale back to the original image.
    const float scale_x = static_cast<float>(image.cols) / 640.0f;
    const float scale_y = static_cast<float>(image.rows) / 512.0f;

    const float x1 = (cx - w / 2.0f) * scale_x;
    const float y1 = (cy - h / 2.0f) * scale_y;
    const float x2 = (cx + w / 2.0f) * scale_x;
    const float y2 = (cy + h / 2.0f) * scale_y;

    // ── draw ──
    cv::rectangle(image,
                  cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                  cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                  cv::Scalar(0, 255, 0), 2);

    const std::string label = "drone " + std::to_string(static_cast<int>(best_score * 100)) + "%";
    cv::putText(image, label,
                cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    cv::imwrite(output_path, image);
    std::cout << "saved: " << output_path << "\n";
    return 0;
}
