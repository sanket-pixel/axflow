#pragma once

#include "axflow/device/device.h"
#include "axflow/config/axflow_config.h"

#include <memory>
#include <vector>
#include <string>

namespace axflow {

// loads model.json, allocates buffers, exposes shape + quant info.
// inference comes later.
class Model {
public:
    Model(Device& device, const AxflowConfig& cfg);
    ~Model();

    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    // shape queries — padded NHWC as the chip sees it
    int num_inputs()  const;
    int num_outputs() const;

    std::vector<int> input_shape (int i = 0) const;
    std::vector<int> output_shape(int i)     const;

    // quant params per tensor
    float input_scale     (int i = 0) const;
    int   input_zero_point(int i = 0) const;
    float output_scale     (int i) const;
    int   output_zero_point(int i) const;

    class Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace axflow