#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace axflow {

// writeable handle into a chip-side int8 input buffer.
// produced by Model::get_input(), consumed by Preprocessor.
// borrows storage from Model — do not use after the Model is destroyed.
class InputBuffer {
public:
  InputBuffer(std::string name, std::vector<int> shape,
              std::vector<std::array<size_t, 2>> padding, float scale,
              int zero_point, int8_t *data, std::size_t size)
      : name_(std::move(name)), shape_(std::move(shape)),
        padding_(std::move(padding)), scale_(scale), zero_point_(zero_point),
        data_(data), size_(size) {}

  const std::string &name() const { return name_; }
  const std::vector<int> &shape() const { return shape_; }
  const std::vector<std::array<size_t, 2>> &padding() const { return padding_; }

  float scale() const { return scale_; }
  int zero_point() const { return zero_point_; }

  int8_t *data() { return data_; }
  std::size_t size() const { return size_; }

private:
  std::string name_;
  std::vector<int> shape_; // padded NHWC
  std::vector<std::array<std::size_t, 2>>
      padding_; // padding_[dim] = {before, after}
  float scale_ = 1.0f;
  int zero_point_ = 0;

  int8_t *data_ = nullptr; // borrowed from Model
  std::size_t size_ = 0;
};

} // namespace axflow