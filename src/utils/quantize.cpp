#include "axflow/utils/quantize.h"

#include <algorithm>
#include <cmath>

namespace axflow
{
    int8_t quantize(float value, float scale, int zero_point)
    {
        const float q = std::round(value / scale + zero_point);
        return static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
    }
} // namespace axflow
