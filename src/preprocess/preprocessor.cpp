#include "axflow/preprocess/preprocessor.h"
#include "axflow/utils/quantize.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace axflow
{
    namespace
    {
        constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
        constexpr float kImagenetStd[3] = {0.229f, 0.224f, 0.225f};
    }

    Preprocessor::Preprocessor(const PreprocessingConfig& cfg) : cfg_(cfg)
    {
    }

    void Preprocessor::run(const cv::Mat& image, InputBuffer& buffer) const
    {
        // ── validate buffer ──
        const auto& shape = buffer.shape();
        const auto& padding = buffer.padding();

        if (shape.size() != 4)
        {
            throw std::runtime_error("axflow::Preprocessor: buffer must be 4D NHWC, got "
                + std::to_string(shape.size()) + "D");
        }
        if (shape[0] - padding[0][0] - padding[0][1] != 1)
        {
            throw std::runtime_error("axflow::Preprocessor: only batch size 1 supported");
        }

        const int padded_h = static_cast<int>(shape[1]);
        const int padded_w = static_cast<int>(shape[2]);
        const int padded_c = static_cast<int>(shape[3]);

        const int unpadded_h = padded_h - static_cast<int>(padding[1][0] + padding[1][1]);
        const int unpadded_w = padded_w - static_cast<int>(padding[2][0] + padding[2][1]);
        const int unpadded_c = padded_c - static_cast<int>(padding[3][0] + padding[3][1]);

        const int pad_h = static_cast<int>(padding[1][0]);
        const int pad_w = static_cast<int>(padding[2][0]);
        const int pad_c = static_cast<int>(padding[3][0]);

        if (unpadded_h <= 0 || unpadded_w <= 0 || unpadded_c <= 0)
        {
            throw std::runtime_error("axflow::Preprocessor: invalid unpadded shape");
        }
        if (unpadded_c != 3)
        {
            throw std::runtime_error("axflow::Preprocessor: only 3-channel inputs supported, got C="
                + std::to_string(unpadded_c));
        }

        // ── validate image ──
        if (image.empty())
        {
            throw std::runtime_error("axflow::Preprocessor: input image is empty");
        }
        if (image.channels() != 3)
        {
            throw std::runtime_error("axflow::Preprocessor: input image must be 3-channel, got "
                + std::to_string(image.channels()));
        }
        if (image.type() != CV_8UC3)
        {
            throw std::runtime_error("axflow::Preprocessor: input image must be CV_8UC3 (uint8 BGR)");
        }

        // ── resize ──
        if (cfg_.resize_mode != "stretch")
        {
            // letterbox not yet implemented — fall back with a warning would go through a logger;
            // for now we just stretch to keep behavior predictable.
        }
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(unpadded_w, unpadded_h));

        // ── BGR → RGB if requested ──
        cv::Mat rgb = resized;
        if (cfg_.input_is_bgr)
        {
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        }

        // ── pre-fill entire buffer with zero_point (neutral padding) ──
        const float scale = buffer.scale();
        const int zp = buffer.zero_point();
        const int8_t pad_val = static_cast<int8_t>(std::clamp(zp, -128, 127));

        std::memset(buffer.data(), static_cast<unsigned char>(pad_val), buffer.size());

        // ── write real pixel data ──
        // NHWC layout, N=1: idx = ((h * padded_w) + w) * padded_c + c
        int8_t* out = buffer.data();

        for (int y = 0; y < unpadded_h; ++y)
        {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < unpadded_w; ++x)
            {
                const cv::Vec3b& px = row[x];

                for (int c = 0; c < 3; ++c)
                {
                    float v = px[c] / 255.0f;
                    if (cfg_.normalize)
                    {
                        v = (v - kImagenetMean[c]) / kImagenetStd[c];
                    }
                    const int idx = ((y + pad_h) * padded_w + (x + pad_w)) * padded_c + (c + pad_c);
                    out[idx] = quantize(v, scale, zp);
                }
            }
        }
    }
} // namespace axflow
