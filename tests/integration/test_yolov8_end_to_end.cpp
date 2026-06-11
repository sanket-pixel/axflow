#include "axflow/axflow.h"
#include "axflow/config/axflow_config.h"
#include "axflow/device/device.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>

using namespace axflow;
namespace fs = std::filesystem;

class Yolov8EndToEnd : public ::testing::Test
{
protected:
    std::string fixture_path(const std::string& rel) const
    {
        return std::string(AXFLOW_TEST_FIXTURES) + "/" + rel;
    }

    void SetUp() override
    {
        // Fail gracefully if hardware/artifacts aren't present
        if (!fs::exists(fixture_path("models/yolov8s-ir/yolov8s-ir/1/model.json")))
        {
            GTEST_SKIP() << "YOLOv8s-IR artifacts not found on this hardware. Skipping.";
        }
        if (!fs::exists(fixture_path("data/sample.png")))
        {
            GTEST_SKIP() << "Sample image missing. Skipping.";
        }
    }
};

TEST_F(Yolov8EndToEnd, RunsFullDetectionPipeline)
{
    Device dev;
    auto cfg = AxflowConfig::from_yaml(fixture_path("configs/yolov8_valid.yaml"));
    cfg.inference.model_dir = fixture_path("models/yolov8s-ir/yolov8s-ir/1");

    AxFlow flow(dev, cfg);
    cv::Mat image = cv::imread(fixture_path("data/sample.png"));
    ASSERT_FALSE(image.empty()) << "Failed to load sample image.";

    EXPECT_NO_THROW({
        flow.preprocess(image);
        flow.inference();
        });

    auto tensors = flow.postamble();

    ASSERT_FALSE(tensors.empty()) << "Postamble should produce outputs.";

    const auto& det = tensors[0];
    ASSERT_EQ(det.shape.size(), 3);
    EXPECT_EQ(det.shape[0], 1);
    EXPECT_EQ(det.shape[1], 27200);
    EXPECT_EQ(det.shape[2], 5);
}
