#pragma once

#include "axflow/config/postamble_config.h"
#include "axflow/data_types/tensor.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace axflow
{
    // runs an ONNX postamble graph (CPU via ONNXRuntime).
    //
    // inputs come from Inference::run() — dequantized float NCHW tensors.
    // mapping is by index: chip output i → ORT input i. shape is verified
    // on every run; mismatch throws loud.
    //
    // usage:
    //   Postamble post(cfg, model_dir);
    //   auto out = post.run(inference_outputs);
    //
    class Postamble
    {
    public:
        Postamble(const PostambleConfig& cfg, const std::string& model_dir);
        ~Postamble();

        Postamble(const Postamble&) = delete;
        Postamble& operator=(const Postamble&) = delete;
        Postamble(Postamble&&) noexcept = default;
        Postamble& operator=(Postamble&&) noexcept = default;

        int num_inputs() const { return input_names_.size(); }
        int num_outputs() const { return output_names_.size(); }

        // run the ORT graph. inputs are passed in chip-output order.
        // returns vector<Tensor> indexed in ORT-output order.
        std::vector<Tensor> run(const std::vector<Tensor>& inputs);

        // named/indexed output access (valid after run())
        const Tensor& get_output(const std::string& name) const;
        const Tensor& get_output(int index) const;

        const std::string& input_name(int i = 0) const;
        const std::string& output_name(int i) const;

    private:
        Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "axflow_postamble"};
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_{nullptr};

        // ORT-owned name strings; cached as plain std::string for comparison
        std::vector<Ort::AllocatedStringPtr> input_name_holders_;
        std::vector<Ort::AllocatedStringPtr> output_name_holders_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

        // expected input shapes for sanity check on each run
        std::vector<std::vector<int>> expected_input_shapes_;

        // last run cached for get_output() lookup
        std::vector<Tensor> outputs_;

        int find_output_index(const std::string& name) const;
    };
} // namespace axflow
