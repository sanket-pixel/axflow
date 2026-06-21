#include "axflow/preprocess/preprocessor.h"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <opencv2/dnn.hpp>

namespace axflow {
    // ImageNet constants for optional normalization
    static constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kImagenetStd[3] = {0.229f, 0.224f, 0.225f};

    Preprocessor::Preprocessor(const PreprocessingConfig &cfg)
        : cfg_(cfg) {
    }

    void Preprocessor::run(const cv::Mat &image, Tensor &out_tensor) const {
        if (out_tensor.shape.size() != 4) {
            throw std::runtime_error("axflow::Preprocessor: out_tensor must be 4D");
        }

        const bool is_nchw = (out_tensor.shape[1] == 3);
        const bool is_nhwc = (out_tensor.shape[3] == 3);

        if (!is_nchw && !is_nhwc) {
            throw std::runtime_error("axflow::Preprocessor: unsupported tensor layout");
        }

        const int target_h = is_nchw ? out_tensor.shape[2] : out_tensor.shape[1];
        const int target_w = is_nchw ? out_tensor.shape[3] : out_tensor.shape[2];

        if (image.empty() || image.channels() != 3 || image.type() != CV_8UC3) {
            throw std::runtime_error("axflow::Preprocessor: invalid input image");
        }

        // 1. Resize and Color Convert
        cv::Mat resized, rgb;
        cv::resize(image, resized, cv::Size(target_w, target_h));

        if (cfg_.input_is_bgr) {
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        } else {
            rgb = resized;
        }

        // 2. Vectorized Scale to [0, 1]
        cv::Mat rgb_f;
        rgb.convertTo(rgb_f, CV_32FC3, 1.0f / 255.0f);

        // 3. Vectorized Normalization
        if (cfg_.normalize) {
            cv::subtract(rgb_f, cv::Scalar(kImagenetMean[0], kImagenetMean[1], kImagenetMean[2]), rgb_f);
            cv::divide(rgb_f, cv::Scalar(kImagenetStd[0], kImagenetStd[1], kImagenetStd[2]), rgb_f);
        }

        // Ensure our output tensor is allocated
        out_tensor.data.resize(out_tensor.numel());

        // 4. Zero-Loop Layout Routing
        if (is_nhwc) {
            // cv::Mat is natively NHWC! We can literally just memcpy it over.
            std::memcpy(out_tensor.data.data(), rgb_f.data, out_tensor.numel() * sizeof(float));
        } else {
            // Use OpenCV's highly optimized DNN backend to transpose NHWC -> NCHW
            cv::Mat blob = cv::dnn::blobFromImage(rgb_f);
            std::memcpy(out_tensor.data.data(), blob.data, out_tensor.numel() * sizeof(float));
        }
    }
} // namespace axflow
