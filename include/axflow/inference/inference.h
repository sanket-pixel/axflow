#pragma once

#include "axflow/config/axflow_config.h"
#include "axflow/data_types/input_buffer.h"
#include "axflow/data_types/tensor.h"
#include "axflow/device/device.h"
#include "axruntime/axruntime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace axflow
{
  // loads model.json, manages chip-side buffers, runs inference.
  //
  // usage:
  //   Model model(device, cfg);
  //   auto in = model.get_input("image");
  //   preprocessor.run(image, in);
  //   auto out = model.run();
  //
  class Inference
  {
  public:
    Inference(Device& device, const AxflowConfig& cfg);
    ~Inference();

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;
    Inference(Inference&&) noexcept = default;
    Inference& operator=(Inference&&) noexcept = default;

    // shape queries
    int num_inputs() const { return input_infos_.size(); }
    int num_outputs() const { return output_infos_.size(); }

    // named input access for preprocessor
    InputBuffer get_input(const std::string& name);
    InputBuffer get_input(int index = 0);

    // run AIPU + dequantize int8 NHWC → float NCHW + strip padding.
    // returns vector indexed in chip-order; also accessible later via
    // get_output().
    std::vector<Tensor> run();

    // named/indexed output access (valid after run())
    const Tensor& get_output(const std::string& name) const;
    const Tensor& get_output(int index) const;

    // raw introspection — useful for tests and debugging
    const std::string& input_name(int i = 0) const;
    const std::string& output_name(int i) const;

  private:
    // borrowed from Device
    axrContext* context_ = nullptr;
    axrConnection* connection_ = nullptr;

    // owned
    axrModel* model_ = nullptr;
    axrModelInstance* instance_ = nullptr;

    // cached at load time
    std::vector<axrTensorInfo> input_infos_;
    std::vector<axrTensorInfo> output_infos_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // pre-allocated chip-side int8 storage
    std::vector<std::vector<int8_t>> input_buffers_;
    std::vector<std::vector<int8_t>> output_buffers_;

    // last run's dequantized outputs
    std::vector<Tensor> outputs_;

    // helpers
    int find_input_index(const std::string& name) const;
    int find_output_index(const std::string& name) const;
    InputBuffer make_input_buffer(int index);
  };
} // namespace axflow
