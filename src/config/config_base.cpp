#include "axflow/config/config_base.h"

namespace axflow {

void ConfigBase::from_file(const std::string &path) {
  parse(YAML::LoadFile(path));
}

void ConfigBase::from_node(const YAML::Node &node) { parse(node); }

} // namespace axflow