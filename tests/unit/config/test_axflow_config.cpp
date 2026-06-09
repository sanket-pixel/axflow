#include "axflow/config/axflow_config.h"

#include <gtest/gtest.h>
#include <string>

using namespace axflow;

class AxflowConfigTest : public ::testing::Test
{
protected:
    std::string fixture(const std::string& rel) const
    {
        return std::string(AXFLOW_TEST_FIXTURES) + "/configs/" + rel;
    }
};

// ─── preprocessing block ─────────────────────────────────────

TEST_F(AxflowConfigTest, LoadsPreprocessingFromValidYaml)
{
    auto axflow_config = AxflowConfig::from_yaml(fixture("yolov8_valid.yaml"));

    EXPECT_EQ(axflow_config.preprocessing.resize_mode, "stretch");
    EXPECT_EQ(axflow_config.preprocessing.normalize, false);
    EXPECT_EQ(axflow_config.preprocessing.input_is_bgr, true);
}

TEST_F(AxflowConfigTest, PreprocessingDefaultsWhenMissing)
{
    auto axflow_config = AxflowConfig::from_yaml(fixture("minimal.yaml"));

    EXPECT_EQ(axflow_config.preprocessing.resize_mode, "stretch");
    EXPECT_EQ(axflow_config.preprocessing.normalize, false);
    EXPECT_EQ(axflow_config.preprocessing.input_is_bgr, true);
}

// ─── inference block ─────────────────────────────────────────

TEST_F(AxflowConfigTest, LoadsInferenceFromValidYaml)
{
    auto axflow_config = AxflowConfig::from_yaml(fixture("yolov8_valid.yaml"));

    EXPECT_EQ(axflow_config.inference.model_dir, "yolov8s-ir/1");
    EXPECT_EQ(axflow_config.inference.num_cores, 1);
}

TEST_F(AxflowConfigTest, InferenceDefaultsWhenMissing)
{
    auto axflow_config = AxflowConfig::from_yaml(fixture("minimal.yaml"));

    EXPECT_EQ(axflow_config.inference.model_dir, "model");
    EXPECT_EQ(axflow_config.inference.num_cores, 1);
}

// ─── full pipeline ───────────────────────────────────────────

TEST_F(AxflowConfigTest, LoadsAllThreeBlocksFromValidYaml)
{
    auto axflow_config = AxflowConfig::from_yaml(fixture("yolov8_valid.yaml"));

    // sanity check: all three sub-configs populated as expected
    EXPECT_EQ(axflow_config.preprocessing.resize_mode, "stretch");
    EXPECT_EQ(axflow_config.inference.model_dir, "yolov8s-ir/1");
    // postprocessing has no fields yet — just confirm it exists with defaults
    // (add assertions here when PostprocessingConfig grows fields)
}

// ─── error cases ─────────────────────────────────────────────

TEST_F(AxflowConfigTest, ThrowsOnMissingFile)
{
    EXPECT_THROW(
        AxflowConfig::from_yaml(fixture("does_not_exist.yaml")),
        YAML::BadFile
    );
}
