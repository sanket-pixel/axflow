#include "axflow/inference/onnx_inference.h"
#include "axflow/preprocess/preprocessor.h"
#include "axflow/config/axflow_config.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>

using namespace axflow;
namespace fs = std::filesystem;

class OnnxInferenceEndToEnd : public ::testing::Test {
protected:
    std::string asset_path(const std::string &rel) const {
        return std::string(AXFLOW_ASSETS_DIR) + "/" + rel;
    }

    void SetUp() override {
        if (!fs::exists(asset_path("models/detection/yolov8s.onnx")) ||
            !fs::exists(asset_path("images/bus.jpg"))) {
            GTEST_SKIP() << "Assets missing. Skipping ONNX test.";
        }
    }
};

TEST_F(OnnxInferenceEndToEnd, RunsPureCpuOnnxPipeline) {
    // 1. Config & Setup (Overrides path just like AIPU test)
    auto cfg = AxflowConfig::from_yaml(asset_path("configs/yolov8_onnx.yaml"));
    cfg.onnx_inference.model_path = asset_path("models/detection/yolov8s.onnx");

    OnnxInference engine(cfg.onnx_inference);
    Preprocessor preprocessor(cfg.preprocessing);

    cv::Mat image = cv::imread(asset_path("images/bus.jpg"));
    ASSERT_FALSE(image.empty()) << "Failed to load sample image.";

    // 2. Execute
    std::vector<Tensor> outputs;
    EXPECT_NO_THROW({
        Tensor input = engine.get_input_tensor("images");
        preprocessor.run(image, input);
        outputs = engine.run({input});
        });

    // 3. Verify Output
    ASSERT_FALSE(outputs.empty());

    const auto &det = outputs[0];
    ASSERT_EQ(det.shape.size(), 3);
    EXPECT_EQ(det.shape[0], 1);
    EXPECT_EQ(det.shape[1], 84);
    EXPECT_EQ(det.shape[2], 8400);
}
