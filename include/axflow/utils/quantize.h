#pragma once

#include <cstdint>

namespace axflow
{
    // asymmetric INT8 quantization for a single value.
    //
    //   q = clamp(round(value / scale + zero_point), -128, 127)
    //
    // matches the convention used by axruntime quantize_params.
    int8_t quantize(float value, float scale, int zero_point);
} // namespace axflow
