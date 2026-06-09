#pragma once

#include "axruntime/axruntime.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace axflow
{
    // extract dims as a plain vector. shape order matches axruntime's reported layout (NHWC).
    std::vector<int> shape_of(const axrTensorInfo& info);

    // extract per-dim padding as {before, after} pairs.
    // padding_of(info)[d] = {info.padding[d][0], info.padding[d][1]}
    std::vector<std::array<std::size_t, 2>> padding_of(const axrTensorInfo& info);

    // shape with padding stripped. throws if any unpadded dim is <= 0.
    std::vector<int> unpadded_shape(const axrTensorInfo& info);
} // namespace axflow
