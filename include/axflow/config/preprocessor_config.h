#pragma once

#include <axflow/config/config_base.h>
#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow {
class PreprocessingConfig : public ConfigBase {
public:
  std::string resize_mode = "stretch"; // stretch | letterbox
  bool normalize = false;              // true = imagenet mean/std
  bool input_is_bgr = true;

protected:
  void parse(const YAML::Node &node) override;
};

} // namespace axflow