#include "axflow/config/preprocessor_config.h"

namespace axflow {
void PreprocessingConfig::parse(const YAML::Node &node) {
  resize_mode = read(node, "resize_mode", resize_mode);
  normalize = read(node, "normalize", normalize);
  input_is_bgr = read(node, "input_is_bgr", input_is_bgr);
}
} // namespace axflow