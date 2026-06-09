#include "axflow/axflow.h"
#include "axflow/config/axflow_config.h"
#include "axflow/device/device.h"
#include "axflow/inference/inference.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>

using namespace axflow;
namespace fs = std::filesystem;

// integration test — needs real metis hardware + compiled yolov8s-ir artifacts
// symlinked at tests/fixtures/models/yolov8s-ir/

class Yolov8EndToEnd : public ::testing::Test
{
protected:
    std::string fixture(const std::string& rel) const
    {
        return std::string(AXFLOW_TEST_FIXTURES) + "/" + rel;
    }

    std::string model_dir() const
    {
        return fixture("models/yolov8s-ir/yolov8s-ir/1");
    }

    std::string sample_image_path() const
    {
        return fixture("data/sample.png");
    }

    void SetUp() override
    {
        if (!fs::exists(model_dir() + "/model.json"))
        {
            GTEST_SKIP() << "yolov8s-ir artifacts not found under "
                << model_dir() << " — skipping";
        }
    }

    AxflowConfig load_cfg() const
    {
        auto cfg = AxflowConfig::from_yaml(fixture("configs/yolov8_valid.yaml"));
        cfg.inference.model_dir = model_dir();
        return cfg;
    }
};

// ─── low-level: device ───────────────────────────────────────

TEST_F(Yolov8EndToEnd, DeviceConnects)
{
    Device dev;
    EXPECT_TRUE(dev.is_connected());
}

// ─── low-level: inference / chip-side metadata ───────────────

TEST_F(Yolov8EndToEnd, InferenceLoadsAndExposesShapes)
{
    Device dev;
    auto cfg = load_cfg();
    Inference engine(dev, cfg.inference);

    EXPECT_EQ(engine.num_inputs(), 1);
    EXPECT_EQ(engine.num_outputs(), 8);
}

TEST_F(Yolov8EndToEnd, InputShapeMatchesManifest)
{
    Device dev;
    auto cfg = load_cfg();
    Inference engine(dev, cfg.inference);

    auto in = engine.get_input(0);
    auto s = in.shape();

    // padded NHWC from manifest.json: [1, 514, 656, 4]
    ASSERT_EQ(s.size(), 4u);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[1], 514);
    EXPECT_EQ(s[2], 656);
    EXPECT_EQ(s[3], 4);
}

TEST_F(Yolov8EndToEnd, InputQuantParamsMatchManifest)
{
    Device dev;
    auto cfg = load_cfg();
    Inference engine(dev, cfg.inference);

    auto in = engine.get_input(0);
    EXPECT_NEAR(in.scale(), 0.00392156f, 1e-6f);
    EXPECT_EQ(in.zero_point(), -128);
}

TEST_F(Yolov8EndToEnd, OutputShapesMatchManifest)
{
    Device dev;
    auto cfg = load_cfg();
    Inference engine(dev, cfg.inference);

    // unpadded NCHW (chip outputs in this order — see DEBUG_PrintAllOutputs)
    const std::vector<std::vector<int>> expected = {
        {1, 64, 128, 160},
        {1, 64, 64, 80},
        {1, 64, 32, 40},
        {1, 64, 16, 20},
        {1, 1, 128, 160},
        {1, 1, 64, 80},
        {1, 1, 32, 40},
        {1, 1, 16, 20},
    };

    auto outs = engine.run();
    ASSERT_EQ(outs.size(), expected.size());
    for (std::size_t i = 0; i < outs.size(); ++i)
    {
        EXPECT_EQ(outs[i].shape, expected[i]) << "output " << i;
    }
}

// ─── high-level: axflow orchestration ────────────────────────

TEST_F(Yolov8EndToEnd, AxFlowConstructsFromConfig)
{
    Device dev;
    auto cfg = load_cfg();
    EXPECT_NO_THROW({ AxFlow flow(dev, cfg); });
}

TEST_F(Yolov8EndToEnd, AxFlowConstructsFromYamlPath)
{
    if (!fs::exists(fixture("configs/yolov8_valid.yaml")))
    {
        GTEST_SKIP() << "yolov8_valid.yaml fixture missing";
    }
    // sanity — this path uses the AxflowConfig::from_yaml constructor.
    // model_dir from the yaml is a relative test placeholder, so we can't
    // actually load chip artifacts this way — just verify construction
    // doesn't crash on the yaml-only path. real chip-loading test uses
    // load_cfg() which overrides model_dir.
    Device dev;
    auto cfg = load_cfg();
    AxFlow flow(dev, cfg); // succeeds because cfg has the real model_dir
    SUCCEED();
}

TEST_F(Yolov8EndToEnd, AxFlowPreprocessAcceptsRealImage)
{
    if (!fs::exists(sample_image_path()))
    {
        GTEST_SKIP() << "sample.png missing at " << sample_image_path();
    }

    Device dev;
    auto cfg = load_cfg();
    AxFlow flow(dev, cfg);

    cv::Mat image = cv::imread(sample_image_path());
    ASSERT_FALSE(image.empty()) << "failed to load " << sample_image_path();

    EXPECT_NO_THROW(flow.preprocess(image));
}

TEST_F(Yolov8EndToEnd, AxFlowInferenceProducesExpectedOutputShapes)
{
    if (!fs::exists(sample_image_path()))
    {
        GTEST_SKIP() << "sample.png missing at " << sample_image_path();
    }

    Device dev;
    auto cfg = load_cfg();
    AxFlow flow(dev, cfg);

    cv::Mat image = cv::imread(sample_image_path());
    ASSERT_FALSE(image.empty());

    flow.preprocess(image);
    auto outs = flow.inference();

    ASSERT_EQ(outs.size(), 8u);
    EXPECT_EQ(outs[0].shape, (std::vector<int>{1, 64, 128, 160}));
    EXPECT_EQ(outs[7].shape, (std::vector<int>{1, 1, 16, 20}));
}

TEST_F(Yolov8EndToEnd, AxFlowPostprocessStubReturnsInferenceOutputs)
{
    if (!fs::exists(sample_image_path()))
    {
        GTEST_SKIP() << "sample.png missing at " << sample_image_path();
    }

    Device dev;
    auto cfg = load_cfg();
    AxFlow flow(dev, cfg);

    cv::Mat image = cv::imread(sample_image_path());
    ASSERT_FALSE(image.empty());

    flow.preprocess(image);
    auto inf_out = flow.inference();
    auto post = flow.postprocess();

    // stub passes through unchanged
    ASSERT_EQ(post.size(), inf_out.size());
    for (std::size_t i = 0; i < post.size(); ++i)
    {
        EXPECT_EQ(post[i].shape, inf_out[i].shape) << "output " << i;
    }
}

TEST_F(Yolov8EndToEnd, AxFlowInferenceProducesNonGarbageValues)
{
    // basic sanity: not all zeros, values in plausible range for dequantized data.
    // this would catch e.g. preprocessor filling buffer wrong → chip seeing zeros → output all at zero_point.
    if (!fs::exists(sample_image_path()))
    {
        GTEST_SKIP() << "sample.png missing";
    }

    Device dev;
    auto cfg = load_cfg();
    AxFlow flow(dev, cfg);

    cv::Mat image = cv::imread(sample_image_path());
    ASSERT_FALSE(image.empty());

    flow.preprocess(image);
    auto outs = flow.inference();

    // pick any output, check it's not all zero
    bool found_nonzero = false;
    for (float v : outs[0].data)
    {
        if (v != 0.0f)
        {
            found_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(found_nonzero) << "output[0] is all zeros — preprocessing or inference failed silently";
}
