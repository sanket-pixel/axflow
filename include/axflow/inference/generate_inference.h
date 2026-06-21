#pragma once

#include "axflow/config/axflow_config.h"
#include "axflow/device/device.h"
#include "axflow/inference/inference_interface.h"

#include <memory>

namespace axflow {
    // Factory function to generate the correct inference engine based on config
    std::unique_ptr<InferenceInterface> generate_inference(Device &device, const AxflowConfig &config);
} // namespace axflow
