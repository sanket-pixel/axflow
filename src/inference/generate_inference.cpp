#include "axflow/inference/generate_inference.h"
#include "axflow/inference/axruntime_inference.h"
#include "axflow/inference/onnx_inference.h"

#include <stdexcept>

namespace axflow {
    std::unique_ptr<InferenceInterface> generate_inference(Device &device, const AxflowConfig &config) {
        if (config.inference_type == "onnx") {
            return std::make_unique<OnnxInference>(config.onnx_inference);
        } else {
            throw std::runtime_error("axflow::generate_inference: unknown inference_type '" +
                                     config.inference_type + "'. Expected 'onnx' or 'axruntime'.");
        }
    }
} // namespace axflow
