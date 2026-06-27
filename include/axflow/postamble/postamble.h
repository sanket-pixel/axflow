#pragma once

#include "axflow/config/postamble_config.h"
#include "axflow/data_types/tensor.h"
#include "axflow/inference/onnx_inference.h"

#include <memory>
#include <string>
#include <vector>

namespace axflow {
    // runs the ORT postamble graph emitted by the Voyager compiler alongside
    // the chip artifacts.
    //
    // thin wrapper over OnnxInference. responsibilities:
    //   - resolve graph_path relative to the chip model_dir
    //   - shape-check AIPU outputs against ORT input slots before running
    //   - delegate the actual ORT execution to OnnxInference
    //
    // inputs are mapped by index (AIPU output i → ORT input i).
    class Postamble {
    public:
        Postamble(const PostambleConfig &config, const std::string &model_dir);

        ~Postamble();

        Postamble(const Postamble &) = delete;

        Postamble &operator=(const Postamble &) = delete;

        Postamble(Postamble &&) noexcept = default;

        Postamble &operator=(Postamble &&) noexcept = default;

        // run the postamble graph. inputs are AIPU dequantized outputs.
        // returns the ORT graph's outputs (e.g. decoded anchors for YOLOv8).
        std::vector<Tensor> &run(const std::vector<Tensor> &aipu_outputs);

    private:
        std::unique_ptr<OnnxInference> engine_;
    };
} // namespace axflow
