#include <axflow/config/axflow_config.h>

namespace axflow
{
  void AxflowConfig::parse(const YAML::Node& node)
  {
    if (node["preprocessing"])
      preprocessing.from_node(node["preprocessing"]);

    if (node["inference"])
      inference.from_node(node["inference"]);

    if (node["postprocessing"])
      postprocessing.from_node(node["postprocessing"]);
  }

  AxflowConfig AxflowConfig::from_yaml(const std::string& path)
  {
    AxflowConfig cfg;
    cfg.from_file(path);
    return cfg;
  }
} // namespace axflow
