#include "axflow/device/device.h"
#include "axflow/model/model.h"
#include "axflow/config/axflow_config.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <string>

using namespace axflow;

namespace fs = std::filesystem;

// integration test — needs real metis hardware + compiled yolov8s-ir artifacts
// symlinked at tests/fixtures/models/yolov8s-ir/

class Yolov8EndToEnd : public ::testing::Test {
protected:
    std::string fixture(const std::string& rel) const {
        return std::string(AXFLOW_TEST_FIXTURES) + "/" + rel;
    }

    std::string model_dir() const {
        return fixture("models/yolov8s-ir/yolov8s-ir/1");
    }

    // skip the whole test if model artifacts are missing
    void SetUp() override {
        if (!fs::exists(model_dir() + "/model.json")) {
            GTEST_SKIP() << "yolov8s-ir artifacts not found under "
                         << model_dir() << " — skipping";
        }
    }

    AxflowConfig load_cfg() const {
        auto cfg = AxflowConfig::from_yaml(fixture("configs/yolov8_valid.yaml"));
        cfg.model_dir = model_dir();
        return cfg;
    }
};

TEST_F(Yolov8EndToEnd, DeviceConnects) {
    Device dev;
    EXPECT_TRUE(dev.is_connected());
}

TEST_F(Yolov8EndToEnd, ModelLoadsAndExposesShapes) {
    Device dev;
    Model m(dev, load_cfg());

    EXPECT_EQ(m.num_inputs(),  1);
    EXPECT_EQ(m.num_outputs(), 8);
}

TEST_F(Yolov8EndToEnd, InputShapeMatchesManifest) {
    Device dev;
    Model m(dev, load_cfg());
    auto s = m.input_shape(0);

    // padded NHWC from manifest.json: [1, 514, 656, 4]
    ASSERT_EQ(s.size(), 4u);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[1], 514);
    EXPECT_EQ(s[2], 656);
    EXPECT_EQ(s[3], 4);
}

TEST_F(Yolov8EndToEnd, InputQuantParamsMatchManifest) {
    Device dev;
    Model m(dev, load_cfg());

    // manifest: quantize_params [[0.003921568859368563, -128]]
    EXPECT_NEAR(m.input_scale(0),       0.00392156f, 1e-6f);
    EXPECT_EQ  (m.input_zero_point(0), -128);
}

TEST_F(Yolov8EndToEnd, OutputShapesMatchManifest) {
    Device dev;
    Model m(dev, load_cfg());

    const std::vector<std::vector<int>> expected = {
        {1, 128, 160, 64},
        {1,  64,  80, 64},
        {1,  32,  40, 64},
        {1,  16,  20, 64},
        {1, 128, 160, 64},
        {1,  64,  80, 64},
        {1,  32,  40, 64},
        {1,  16,  20, 64},
    };

    ASSERT_EQ(m.num_outputs(), static_cast<int>(expected.size()));
    for (int i = 0; i < m.num_outputs(); ++i) {
        EXPECT_EQ(m.output_shape(i), expected[i]) << "output " << i;
    }
}

TEST_F(Yolov8EndToEnd, OutputQuantParamsMatchManifest) {
    Device dev;
    Model m(dev, load_cfg());

    // dequantize_params from manifest.json (scale, zero_point per output)
    const std::vector<std::pair<float,int>> expected = {
        {0.16340677f,  27},
        {0.28699985f,  84},
        {0.12218219f,  35},
        {0.09848355f,   0},
        {0.91475081f, 111},
        {0.86987889f, 117},
        {1.45489192f, 124},
        {2.34431243f, 126},
    };

    for (int i = 0; i < m.num_outputs(); ++i) {
        EXPECT_NEAR(m.output_scale(i),      expected[i].first,  1e-5f) << "output " << i;
        EXPECT_EQ  (m.output_zero_point(i), expected[i].second)        << "output " << i;
    }
}