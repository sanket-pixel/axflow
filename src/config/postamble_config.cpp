#include "axflow/config/postamble_config.h"

namespace axflow
{
    void PostambleConfig::parse(const YAML::Node& node)
    {
        enabled = read(node, "enabled", enabled);
        graph_path = read(node, "graph_path", graph_path);
        intra_op_num_threads = read(node, "intra_op_num_threads", intra_op_num_threads);
    }
} // namespace axflow
