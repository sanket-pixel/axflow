#pragma once

#include "axflow/config/config_base.h"
#include "axflow/config/axruntime_inference_config.h"
#include "axflow/config/onnx_inference_config.h"
#include "axflow/config/preprocessor_config.h"
#include "axflow/config/postamble_config.h"
#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow {
    // top-level config — three pipeline stages.
    //
    // yaml shape:
    //   inference_type: "onnx" # or "axruntime"
    //   preprocessing:
    //     ...
    //   inference:
    //     model_dir: ...
    //     num_cores: 1
    //   onnx_inference:
    //     model_path: ...
    //   postamble:
    //     ...
    //
    class AxflowConfig : public ConfigBase {
    public:
        std::string inference_type = "axruntime"; // Default to native hardware

        PreprocessingConfig preprocessing;
        AxruntimeInferenceConfig axruntime_inference;
        OnnxInferenceConfig onnx_inference;
        PostambleConfig postamble;

        // load everything from one yaml file
        static AxflowConfig from_yaml(const std::string &path);

    protected:
        void parse(const YAML::Node &node) override;
    };
} // namespace axflow
