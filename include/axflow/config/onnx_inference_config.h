// include/axflow/config/onnx_inference_config.h
#pragma once

#include "axflow/config/config_base.h"

#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow
{
    // pure ORT inference block — points to an uncompiled onnx model
    class OnnxInferenceConfig : public ConfigBase
    {
    public:
        std::string model_path{"model.onnx"};
        int intra_op_num_threads = 4;

    protected:
        void parse(const YAML::Node& node) override;
    };
} // namespace axflow