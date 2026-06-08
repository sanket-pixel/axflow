#include "axflow/config/axflow_config.h"
#include <gtest/gtest.h>
#include <string>

using namespace axflow;
// base fixture — all config tests inherit this
class AxflowConfigTest : public ::testing::Test {
protected:
  std::string fixture(const std::string &rel) const {
    return std::string(AXFLOW_TEST_FIXTURES) + "/configs/" + rel;
  }
};

// valid full config loads correctly
TEST_F(AxflowConfigTest, LoadsValidYaml) {
  auto cfg = AxflowConfig::from_yaml(fixture("yolov8_valid.yaml"));

  EXPECT_EQ(cfg.model_dir, "yolov8s-ir/1");
  EXPECT_EQ(cfg.num_cores, 1);
}

// preprocessing block parses correctly
TEST_F(AxflowConfigTest, LoadsPreprocessing) {
  auto cfg = AxflowConfig::from_yaml(fixture("yolov8_valid.yaml"));

  EXPECT_EQ(cfg.preprocessing.resize_mode, "stretch");
  EXPECT_EQ(cfg.preprocessing.normalize, false);
  EXPECT_EQ(cfg.preprocessing.input_is_bgr, true);
}

// minimal yaml — missing fields fall back to defaults
TEST_F(AxflowConfigTest, DefaultsOnMinimalYaml) {
  auto cfg = AxflowConfig::from_yaml(fixture("minimal.yaml"));

  EXPECT_EQ(cfg.model_dir, "model");
  EXPECT_EQ(cfg.num_cores, 1);                         // default
  EXPECT_EQ(cfg.preprocessing.normalize, false);       // default
  EXPECT_EQ(cfg.preprocessing.input_is_bgr, true);     // default
  EXPECT_EQ(cfg.preprocessing.resize_mode, "stretch"); // default
}

// missing file throws
TEST_F(AxflowConfigTest, ThrowsOnMissingFile) {
  EXPECT_THROW(AxflowConfig::from_yaml(fixture("does_not_exist.yaml")),
               YAML::BadFile);
}