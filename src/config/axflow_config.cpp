#include "axflow/config/axflow_config.h"

namespace axflow {

void AxflowConfig::parse(const YAML::Node &node) {
  auto root = node["axflow"] ? node["axflow"] : node;
  model_dir = read(root, "model_dir", model_dir);
  num_cores = read(root, "num_cores", num_cores);
  if (node["preprocessing"])
    preprocessing.from_node(node["preprocessing"]);
}

AxflowConfig AxflowConfig::from_yaml(const std::string &path) {
  AxflowConfig cfg;
  cfg.from_file(path);
  return cfg;
}

} // namespace axflow