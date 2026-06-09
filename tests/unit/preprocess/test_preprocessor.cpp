#include "axflow/config/preprocessor_config.h"
#include "axflow/data_types/input_buffer.h"
#include "axflow/preprocess/preprocessor.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace axflow;
namespace fs = std::filesystem;

// helper — build a synthetic InputBuffer backed by a vector we own and can inspect.
class FakeBuffer
{
public:
    FakeBuffer(std::size_t padded_h, std::size_t padded_w, std::size_t padded_c,
               std::size_t pad_h_top, std::size_t pad_h_bot,
               std::size_t pad_w_left, std::size_t pad_w_right,
               std::size_t pad_c_start, std::size_t pad_c_end,
               float scale, int zero_point)
        : storage_(padded_h * padded_w * padded_c, int8_t(0)),
          shape_({
              1, static_cast<int>(padded_h),
              static_cast<int>(padded_w),
              static_cast<int>(padded_c)
          }),
          padding_({
              {
                  {0, 0},
                  {pad_h_top, pad_h_bot},
                  {pad_w_left, pad_w_right},
                  {pad_c_start, pad_c_end}
              }
          }),
          scale_(scale), zero_point_(zero_point)
    {
    }

    InputBuffer view()
    {
        return InputBuffer("test_input", shape_, padding_,
                           scale_, zero_point_,
                           storage_.data(), storage_.size());
    }

    const std::vector<int8_t>& storage() const { return storage_; }

    int8_t at(int h, int w, int c) const
    {
        const int padded_w = shape_[2];
        const int padded_c = shape_[3];
        return storage_[(h * padded_w + w) * padded_c + c];
    }

private:
    std::vector<int8_t> storage_;
    std::vector<int> shape_;
    std::vector<std::array<std::size_t, 2>> padding_;
    float scale_;
    int zero_point_;
};

// shared defaults — YOLO IR style quant params
static PreprocessingConfig basic_cfg()
{
    PreprocessingConfig c;
    c.resize_mode = "stretch";
    c.normalize = false;
    c.input_is_bgr = true;
    return c;
}

// ─── correctness on synthetic input ──────────────────────────────────

TEST(PreprocessorTest, FillsPaddedRegionsWithZeroPoint)
{
    // padded [1, 12, 16, 4], pad: H={1,1}, W={2,2}, C={0,1}
    // unpadded H=10, W=12, C=3
    FakeBuffer fb(12, 16, 4, 1, 1, 2, 2, 0, 1,
                  0.00392156f, -128);
    auto in = fb.view();

    cv::Mat image(20, 30, CV_8UC3, cv::Scalar(255, 255, 255));

    Preprocessor pp(basic_cfg());
    pp.run(image, in);

    // top padding row
    for (int w = 0; w < 16; ++w)
        for (int c = 0; c < 4; ++c)
            EXPECT_EQ(fb.at(0, w, c), int8_t(-128))
                << "top pad: h=0 w=" << w << " c=" << c;

    // bottom padding row
    for (int w = 0; w < 16; ++w)
        for (int c = 0; c < 4; ++c)
            EXPECT_EQ(fb.at(11, w, c), int8_t(-128))
                << "bottom pad: h=11 w=" << w << " c=" << c;

    // left padding cols
    for (int h = 0; h < 12; ++h)
        for (int w = 0; w < 2; ++w)
            for (int c = 0; c < 4; ++c)
                EXPECT_EQ(fb.at(h, w, c), int8_t(-128))
                    << "left pad: h=" << h << " w=" << w << " c=" << c;

    // padded channel slot
    for (int h = 0; h < 12; ++h)
        for (int w = 0; w < 16; ++w)
            EXPECT_EQ(fb.at(h, w, 3), int8_t(-128))
                << "pad channel: h=" << h << " w=" << w;
}

TEST(PreprocessorTest, QuantizesKnownPixelCorrectly)
{
    FakeBuffer fb(1, 1, 3, 0, 0, 0, 0, 0, 0,
                  0.00392156f, -128);
    auto in = fb.view();

    cv::Mat image(1, 1, CV_8UC3, cv::Scalar(0, 128, 255)); // BGR

    Preprocessor pp(basic_cfg());
    pp.run(image, in);

    // BGR (0,128,255) → RGB (255,128,0)
    // q = round(v/255 / 0.00392156 - 128)
    EXPECT_EQ(fb.at(0, 0, 0), int8_t( 127));
    EXPECT_EQ(fb.at(0, 0, 1), int8_t( 0));
    EXPECT_EQ(fb.at(0, 0, 2), int8_t(-128));
}

TEST(PreprocessorTest, NHWCLayoutIsCorrect)
{
    FakeBuffer fb(4, 4, 4, 1, 1, 1, 1, 0, 1,
                  0.00392156f, -128);
    auto in = fb.view();

    cv::Mat image(2, 2, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(50, 100, 150);
    image.at<cv::Vec3b>(0, 1) = cv::Vec3b(25, 75, 125);
    image.at<cv::Vec3b>(1, 0) = cv::Vec3b(200, 150, 100);
    image.at<cv::Vec3b>(1, 1) = cv::Vec3b(10, 20, 30);

    Preprocessor pp(basic_cfg());
    pp.run(image, in);

    auto expect_pixel = [&](int y, int x, uint8_t R, uint8_t G, uint8_t B)
    {
        const int bh = y + 1, bw = x + 1;
        EXPECT_EQ(fb.at(bh, bw, 0), static_cast<int8_t>(R - 128)) << "y=" << y << " x=" << x << " R";
        EXPECT_EQ(fb.at(bh, bw, 1), static_cast<int8_t>(G - 128)) << "y=" << y << " x=" << x << " G";
        EXPECT_EQ(fb.at(bh, bw, 2), static_cast<int8_t>(B - 128)) << "y=" << y << " x=" << x << " B";
    };

    expect_pixel(0, 0, 150, 100, 50);
    expect_pixel(0, 1, 125, 75, 25);
    expect_pixel(1, 0, 100, 150, 200);
    expect_pixel(1, 1, 30, 20, 10);
}

TEST(PreprocessorTest, ResizeShrinksLargerImage)
{
    FakeBuffer fb(4, 4, 3, 0, 0, 0, 0, 0, 0,
                  0.00392156f, -128);
    auto in = fb.view();

    cv::Mat image(80, 100, CV_8UC3, cv::Scalar(128, 128, 128));

    Preprocessor pp(basic_cfg());
    pp.run(image, in);

    for (int h = 0; h < 4; ++h)
        for (int w = 0; w < 4; ++w)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(fb.at(h, w, c), int8_t(0), 1)
                    << "h=" << h << " w=" << w << " c=" << c;
}

// ─── validation / error handling ─────────────────────────────────────

TEST(PreprocessorTest, ThrowsOnNon3ChannelBuffer)
{
    FakeBuffer fb(2, 2, 1, 0, 0, 0, 0, 0, 0, 0.01f, 0);
    auto in = fb.view();
    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));

    Preprocessor pp(basic_cfg());
    EXPECT_THROW(pp.run(image, in), std::runtime_error);
}

TEST(PreprocessorTest, ThrowsOnGrayscaleImage)
{
    FakeBuffer fb(2, 2, 3, 0, 0, 0, 0, 0, 0, 0.01f, 0);
    auto in = fb.view();
    cv::Mat image(2, 2, CV_8UC1, cv::Scalar(0));

    Preprocessor pp(basic_cfg());
    EXPECT_THROW(pp.run(image, in), std::runtime_error);
}

TEST(PreprocessorTest, ThrowsOnEmptyImage)
{
    FakeBuffer fb(2, 2, 3, 0, 0, 0, 0, 0, 0, 0.01f, 0);
    auto in = fb.view();
    cv::Mat empty;

    Preprocessor pp(basic_cfg());
    EXPECT_THROW(pp.run(empty, in), std::runtime_error);
}

// ─── end-to-end with a real image ────────────────────────────────────

TEST(PreprocessorTest, EndToEndWithSamplePng)
{
    const std::string path = std::string(AXFLOW_TEST_FIXTURES) + "/data/sample.png";
    if (!fs::exists(path))
    {
        GTEST_SKIP() << "sample.png not found at " << path;
    }

    cv::Mat image = cv::imread(path);
    ASSERT_FALSE(image.empty()) << "failed to load " << path;

    FakeBuffer fb(514, 656, 4, 1, 1, 8, 8, 0, 1,
                  0.00392156f, -128);
    auto in = fb.view();

    Preprocessor pp(basic_cfg());
    EXPECT_NO_THROW(pp.run(image, in));

    bool found_non_pad = false;
    for (auto v : fb.storage())
    {
        if (v != int8_t(-128))
        {
            found_non_pad = true;
            break;
        }
    }
    EXPECT_TRUE(found_non_pad) << "buffer was not populated";
}
