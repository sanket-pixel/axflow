// src/config/onnx_inference_config.cpp
#include "axflow/config/onnx_inference_config.h"

namespace axflow
{
    void OnnxInferenceConfig::parse(const YAML::Node& node)
    {
        model_path = read(node, "model_path", model_path);
        intra_op_num_threads = read(node, "intra_op_num_threads", intra_op_num_threads);
    }
} // namespace axflow