#pragma once

#include "axflow/data_types/tensor.h"
#include <string>
#include <vector>

namespace axflow {
    class InferenceInterface {
    public:
        virtual ~InferenceInterface() = default;

        // ─── Introspection ───

        virtual int input_count() const = 0;

        virtual int output_count() const = 0;

        // ─── Names (Clang-Tidy Compliant) ───

        virtual const std::string &input_name(int index) const = 0;

        const std::string &input_name() const { return input_name(0); }

        virtual const std::string &output_name(int index) const = 0;

        const std::string &output_name() const { return output_name(0); }

        // ─── Zero-Copy Memory Access ───

        // Exposes the engine's internally allocated float32 input buffer
        virtual Tensor &get_input_tensor(int index) = 0;

        Tensor &get_input_tensor() { return get_input_tensor(0); }

        // ─── Execution ───

        // Executes inference using the internal pre-bound buffers.
        virtual std::vector<Tensor> &run() = 0;
    };
} // namespace axflow
