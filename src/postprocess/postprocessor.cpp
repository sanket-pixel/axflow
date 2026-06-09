#include "axflow/postprocess/postprocessor.h"

namespace axflow
{
    Postprocessor::Postprocessor(const PostprocessingConfig& cfg) : cfg_(cfg)
    {
    }

    std::vector<Tensor> Postprocessor::run(const std::vector<Tensor>& inputs) const
    {
        // stub: pass-through. real ORT/NMS/decoding wiring comes later.
        return inputs;
    }
} // namespace axflow
