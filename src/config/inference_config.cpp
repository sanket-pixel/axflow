#include <axflow/config/inference_config.h>

namespace axflow
{
    void InferenceConfig::parse(const YAML::Node& node)
    {
        model_dir = read(node, "model_dir", model_dir);
        num_cores = read(node, "num_cores", num_cores);
    }
} // namespace axflow
