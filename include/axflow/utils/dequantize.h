#pragma once

#include "axflow/data_types/tensor.h"
#include "axruntime/axruntime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace axflow
{
    // dequantize int8 NHWC (padded, chip layout) → float32 NCHW (dense).
    //
    // assumes:
    //   - info.ndims == 4
    //   - layout is NHWC (only layout supported by AxRuntime today)
    //   - padding may exist on any dim; it is stripped
    //
    // throws for non-4D tensors or invalid unpadded shapes.
    Tensor dequantize_nhwc_to_nchw(const axrTensorInfo& info,
                                   const std::vector<int8_t>& int8_buffer,
                                   const std::string& name);
} // namespace axflow
