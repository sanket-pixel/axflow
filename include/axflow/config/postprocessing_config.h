#pragma once

#include "axflow/config/config_base.h"

#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow
{
    // postprocessing block — currently a stub.
    //
    // future fields will likely include:
    //   type:          none | onnx | custom
    //   graph:         path to postprocess_graph.onnx
    //   conf_threshold, nms_threshold, ...
    //
    class PostprocessingConfig : public ConfigBase
    {
    public:
        std::string type{"none"}; // none | onnx | custom

    protected:
        void parse(const YAML::Node& node) override;
    };
} // namespace axflow
