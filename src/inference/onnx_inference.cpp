#include "axflow/inference/onnx_inference.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace axflow {
    OnnxInference::OnnxInference(const OnnxInferenceConfig &configuration) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(configuration.intra_op_num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
            session_ = std::make_unique<Ort::Session>(env_, configuration.model_path.c_str(), session_options);
        } catch (const std::exception &error) {
            throw std::runtime_error("axflow::OnnxInference: failed to load model from " +
                                     configuration.model_path + " - " + error.what());
        }
        io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;
        // Delegate the heavy lifting to the DRY helper
        initialize_inputs(allocator); // Setup Inputs
        initialize_outputs(allocator); // Setup Outputs
    }

    std::vector<int64_t> OnnxInference::get_sanitized_shape(const Ort::TypeInfo &type_information) const {
        auto tensor_info = type_information.GetTensorTypeAndShapeInfo();
        auto raw_shape = tensor_info.GetShape();

        std::vector<int64_t> clean_shape;
        clean_shape.reserve(raw_shape.size());

        for (int64_t dimension: raw_shape) {
            // Replace dynamic dimensions (<= 0) with 1
            clean_shape.push_back(dimension <= 0 ? 1 : dimension);
        }

        return clean_shape;
    }

    void OnnxInference::initialize_inputs(Ort::AllocatorWithDefaultOptions &allocator) {
        const size_t tensor_count = session_->GetInputCount();

        input_tensors_.resize(tensor_count);
        ort_inputs_.reserve(tensor_count);
        input_names_.reserve(tensor_count);
        input_shapes_.reserve(tensor_count);

        for (size_t index = 0; index < tensor_count; ++index) {
            // 1. Extract and Cache Name
            Ort::AllocatedStringPtr allocated_name = session_->GetInputNameAllocated(index, allocator);
            input_names_.push_back(allocated_name.get());
            input_name_holders_.push_back(std::move(allocated_name));

            // 2. Extract and Cache Shape
            Ort::TypeInfo type_information = session_->GetInputTypeInfo(index);
            std::vector<int64_t> clean_shape64 = get_sanitized_shape(type_information);
            input_shapes_.push_back(clean_shape64);

            // 3. Allocate Physical Memory
            Tensor &current_tensor = input_tensors_[index];
            current_tensor.name = input_names_.back();

            for (int64_t dimension: clean_shape64) {
                current_tensor.shape.push_back(static_cast<int>(dimension));
            }

            current_tensor.data.resize(current_tensor.numel(), 0.0f);

            // 4. Create ORT Wrapper (Must use persistent shapes memory)
            ort_inputs_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                current_tensor.data.data(),
                current_tensor.data.size(),
                input_shapes_.back().data(),
                input_shapes_.back().size()
            ));

            // 5. Bind to the IO Driver
            io_binding_->BindInput(input_names_.back().c_str(), ort_inputs_.back());
        }
    }

    void OnnxInference::initialize_outputs(Ort::AllocatorWithDefaultOptions &allocator) {
        const size_t tensor_count = session_->GetOutputCount();
        output_tensors_.resize(tensor_count);
        ort_outputs_.reserve(tensor_count);
        output_names_.reserve(tensor_count);
        output_shapes_.reserve(tensor_count);
        for (size_t index = 0; index < tensor_count; ++index) {
            // 1. Extract and Cache Name
            Ort::AllocatedStringPtr allocated_name = session_->GetOutputNameAllocated(index, allocator);
            output_names_.push_back(allocated_name.get());
            output_name_holders_.push_back(std::move(allocated_name));
            // 2. Extract and Cache Shape
            Ort::TypeInfo type_information = session_->GetOutputTypeInfo(index);
            std::vector<int64_t> clean_shape64 = get_sanitized_shape(type_information);
            output_shapes_.push_back(clean_shape64);
            // 3. Allocate Physical Memory
            Tensor &current_tensor = output_tensors_[index];
            current_tensor.name = output_names_.back();
            for (int64_t dimension: clean_shape64) {
                current_tensor.shape.push_back(static_cast<int>(dimension));
            }
            current_tensor.data.resize(current_tensor.numel(), 0.0f);
            // 4. Create ORT Wrapper (Must use persistent shapes memory)
            ort_outputs_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                current_tensor.data.data(),
                current_tensor.data.size(),
                output_shapes_.back().data(),
                output_shapes_.back().size()
            ));
            // 5. Bind to the IO Driver
            io_binding_->BindOutput(output_names_.back().c_str(), ort_outputs_.back());
        }
    }

    OnnxInference::~OnnxInference() = default;

    int OnnxInference::find_input_index(const std::string &name) const {
        for (std::size_t i = 0; i < input_names_.size(); ++i) {
            if (input_names_[i] == name) return static_cast<int>(i);
        }
        throw std::runtime_error("axflow::OnnxInference: no input named '" + name + "'");
    }

    int OnnxInference::find_output_index(const std::string &name) const {
        for (std::size_t i = 0; i < output_names_.size(); ++i) {
            if (output_names_[i] == name) return static_cast<int>(i);
        }
        throw std::runtime_error("axflow::OnnxInference: no output named '" + name + "'");
    }

    Tensor &OnnxInference::get_input_tensor(int index) {
        return input_tensors_.at(index);
    }

    Tensor &OnnxInference::get_input_tensor(const std::string &name) {
        return input_tensors_.at(find_input_index(name));
    }

    std::vector<Tensor> &OnnxInference::run() {
        // 1. Execute via IoBinding (Zero copies, zero allocations, zero overhead routing)
        // The Preprocessor already wrote directly into the mapped input_tensors_ memory!
        session_->Run(Ort::RunOptions{nullptr}, *io_binding_);
        // 2. Return pre-allocated outputs (populated directly by the ORT C++ backend)
        return output_tensors_;
    }

    const Tensor &OnnxInference::get_output_tensor(int index) const {
        return output_tensors_.at(index);
    }

    const Tensor &OnnxInference::get_output_tensor(const std::string &name) const {
        return output_tensors_.at(find_output_index(name));
    }

    const std::string &OnnxInference::input_name(int index) const {
        return input_names_.at(index);
    }

    const std::string &OnnxInference::output_name(int index) const {
        return output_names_.at(index);
    }
} // namespace axflow
