#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace axflow {

// dequantized float output tensor.
// owned by Model after run(); references invalidated by the next run().
struct Tensor {
  std::string name;
  std::vector<int> shape;  // NCHW, padding stripped
  std::vector<float> data; // dense, contiguous

  std::size_t numel() const {
    std::size_t n = 1;
    for (int d : shape)
      n *= d;
    return n;
  }

  int dim(int i) const { return shape.at(i); }
};

} // namespace axflow