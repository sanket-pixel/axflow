#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow {

class ConfigBase {
public:
  ConfigBase() = default;
  virtual ~ConfigBase() = default;

  void from_file(const std::string &path);
  void from_node(const YAML::Node &node);

protected:
  // derived classes implement this
  virtual void parse(const YAML::Node &node) = 0;

  // safe read — returns fallback if key missing
  template <typename T>
  static T read(const YAML::Node &node, const std::string &key,
                const T &fallback) {
    if (node[key])
      return node[key].as<T>();
    return fallback;
  }
};
} // namespace axflow