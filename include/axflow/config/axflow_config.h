#pragma once

#include <axflow/config/config_base.h>
#include <axflow/config/preprocessor_config.h>
#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow {

// top-level axflow config
class AxflowConfig : public ConfigBase {
public:
  std::string model_dir{"model"};
  int num_cores = 1;
  PreprocessingConfig preprocessing;

  // load everything from one yaml file
  static AxflowConfig from_yaml(const std::string &path);

protected:
  void parse(const YAML::Node &node) override;
};

} // namespace axflow