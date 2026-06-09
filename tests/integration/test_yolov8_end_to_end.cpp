#include "axflow/config/axflow_config.h"
#include "axflow/device/device.h"
#include "axflow/inference/inference.h"

#include <filesystem>
#include <gtest/gtest.h>
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

  void SetUp() override
  {
    if (!fs::exists(model_dir() + "/model.json"))
    {
      GTEST_SKIP() << "yolov8s-ir artifacts not found under " << model_dir()
        << " — skipping";
    }
  }

  AxflowConfig load_cfg() const
  {
    auto cfg = AxflowConfig::from_yaml(fixture("configs/yolov8_valid.yaml"));
    cfg.inference.model_dir = model_dir();
    return cfg;
  }
};

TEST_F(Yolov8EndToEnd, DeviceConnects)
{
  Device dev;
  EXPECT_TRUE(dev.is_connected());
}

TEST_F(Yolov8EndToEnd, ModelLoadsAndExposesShapes)
{
  Device dev;
  Model m(dev, load_cfg());

  EXPECT_EQ(m.num_inputs(), 1);
  EXPECT_EQ(m.num_outputs(), 8);
}

TEST_F(Yolov8EndToEnd, InputShapeMatchesManifest)
{
  Device dev;
  Model m(dev, load_cfg());
  auto in = m.get_input(0);
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
  Model m(dev, load_cfg());
  auto in = m.get_input(0);

  // manifest: quantize_params [[0.003921568859368563, -128]]
  EXPECT_NEAR(in.scale(), 0.00392156f, 1e-6f);
  EXPECT_EQ(in.zero_point(), -128);
}

TEST_F(Yolov8EndToEnd, OutputShapesMatchManifest)
{
  Device dev;
  Model m(dev, load_cfg());

  // ground truth from chip (DEBUG_PrintAllOutputs):
  //   0-3: box branches  (C=64)
  //   4-7: score branches (C=1, padded from 64)
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

  auto outs = m.run();
  ASSERT_EQ(outs.size(), expected.size());
  for (std::size_t i = 0; i < outs.size(); ++i)
  {
    EXPECT_EQ(outs[i].shape, expected[i]) << "output " << i;
  }
}
