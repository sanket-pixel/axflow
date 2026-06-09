#pragma once

#include <axflow/config/config_base.h>
#include <axflow/config/inference_config.h>
#include <axflow/config/postprocessing_config.h>
#include <axflow/config/preprocessor_config.h>

#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow
{
  // top-level config — three pipeline stages.
  //
  // yaml shape:
  //   preprocessing:
  //     ...
  //   inference:
  //     model_dir: ...
  //     num_cores: 1
  //   postprocessing:
  //     ...
  //
  class AxflowConfig : public ConfigBase
  {
  public:
    PreprocessingConfig preprocessing;
    InferenceConfig inference;
    // PostprocessingConfig  postprocessing; configs

    // load everything from one yaml file
    static AxflowConfig from_yaml(const std::string& path);

  protected:
    void parse(const YAML::Node& node) override;
  };
} // namespace axflow
