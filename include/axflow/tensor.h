#pragma once

#include <vector>
#include <cstddef>

namespace axflow {

// plain dequantized tensor — no axruntime dependency
struct Tensor {
    std::vector<float> data;
    std::vector<int>   shape;   // NCHW after padding stripped

    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= d;
        return n;
    }

    int dim(int i) const { return shape.at(i); }
};

} // namespace axflow