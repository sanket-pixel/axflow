#pragma once

#include "axflow/config/config_base.h"

#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow
{
    // postamble block — runs an ONNX graph emitted by the Voyager compiler
    // alongside the chip artifacts.
    //
    // yaml shape:
    //   postamble:
    //     enabled: true
    //     graph_path: postprocess_graph.onnx    # relative to inference.model_dir
    //     intra_op_num_threads: 4
    //
    class PostambleConfig : public ConfigBase
    {
    public:
        bool enabled = false;
        std::string graph_path = "postprocess_graph.onnx";
        int intra_op_num_threads = 4;

    protected:
        void parse(const YAML::Node& node) override;
    };
} // namespace axflow
