#include "axflow/config/axflow_config.h"

namespace axflow {
  void AxflowConfig::parse(const YAML::Node &node) {
    // 1. Route the factory
    if (node["inference_type"]) {
      inference_type = node["inference_type"].as<std::string>();
    }

    // 2. Parse sub-configs
    if (node["preprocessing"]) {
      preprocessing.from_node(node["preprocessing"]);
    }

    if (node["inference"]) {
      inference.from_node(node["inference"]);
    }

    if (node["onnx_inference"]) {
      onnx_inference.from_node(node["onnx_inference"]);
    }

    if (node["postamble"]) {
      postamble.from_node(node["postamble"]);
    }
  }

  AxflowConfig AxflowConfig::from_yaml(const std::string &path) {
    AxflowConfig cfg;
    cfg.from_file(path);
    return cfg;
  }
} // namespace axflow
