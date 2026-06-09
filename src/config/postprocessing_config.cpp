#include "axflow/config/postprocessing_config.h"

namespace axflow
{
    void PostprocessingConfig::parse(const YAML::Node& node)
    {
        type = read(node, "type", type);
    }
} // namespace axflow
