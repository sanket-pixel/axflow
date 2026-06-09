#pragma once

#include "axflow/config/postprocessing_config.h"
#include "axflow/data_types/tensor.h"

#include <vector>

namespace axflow
{
    // stage 3 of the pipeline.
    //
    // currently a stub: returns inputs unchanged.
    //
    // real implementations will run an ONNX postamble graph, apply NMS,
    // decode bounding boxes, etc. — wired here so AxFlow has a consistent
    // pipeline shape today.
    class Postprocessor
    {
    public:
        explicit Postprocessor(const PostprocessingConfig& cfg);

        // takes the dequantized chip outputs and returns final user-facing tensors.
        // stub behavior: pass-through.
        std::vector<Tensor> run(const std::vector<Tensor>& inputs) const;

    private:
        PostprocessingConfig cfg_;
    };
} // namespace axflow
