#include "axflow/postamble/postamble.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace axflow
{
    namespace
    {
        std::string shape_str(const std::vector<int>& s)
        {
            std::ostringstream os;
            os << "[";
            for (std::size_t i = 0; i < s.size(); ++i)
            {
                os << s[i];
                if (i + 1 < s.size()) os << ",";
            }
            os << "]";
            return os.str();
        }
    } // anonymous namespace

    Postamble::Postamble(const PostambleConfig& cfg, const std::string& model_dir)
    {
        if (!cfg.enabled)
        {
            throw std::runtime_error("axflow::Postamble: constructed but enabled=false in config");
        }

        auto graph_path = std::filesystem::path(model_dir) / cfg.graph_path;
        if (!std::filesystem::exists(graph_path))
        {
            throw std::runtime_error("axflow::Postamble: graph not found at " + graph_path.string());
        }

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(cfg.intra_op_num_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, graph_path.string().c_str(), opts);
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions alloc;

        // cache input names + expected shapes
        const std::size_t n_in = session_->GetInputCount();
        input_name_holders_.reserve(n_in);
        input_names_.reserve(n_in);
        expected_input_shapes_.reserve(n_in);

        for (std::size_t i = 0; i < n_in; ++i)
        {
            auto name_holder = session_->GetInputNameAllocated(i, alloc);
            input_names_.emplace_back(name_holder.get());
            input_name_holders_.push_back(std::move(name_holder));

            auto type_info = session_->GetInputTypeInfo(i);
            auto shape_info = type_info.GetTensorTypeAndShapeInfo();
            auto raw_shape = shape_info.GetShape();

            std::vector<int> shape;
            shape.reserve(raw_shape.size());
            for (auto d : raw_shape) shape.push_back(static_cast<int>(d));
            expected_input_shapes_.push_back(std::move(shape));
        }

        // cache output names
        const std::size_t n_out = session_->GetOutputCount();
        output_name_holders_.reserve(n_out);
        output_names_.reserve(n_out);

        for (std::size_t i = 0; i < n_out; ++i)
        {
            auto name_holder = session_->GetOutputNameAllocated(i, alloc);
            output_names_.emplace_back(name_holder.get());
            output_name_holders_.push_back(std::move(name_holder));
        }
    }

    Postamble::~Postamble() = default;

    std::vector<Tensor> Postamble::run(const std::vector<Tensor>& inputs)
    {
        if (inputs.size() != input_names_.size())
        {
            throw std::runtime_error(
                "axflow::Postamble: expected " + std::to_string(input_names_.size())
                + " inputs, got " + std::to_string(inputs.size()));
        }

        // shape sanity — fail loud if Voyager output order ever drifts from ORT graph order
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            if (inputs[i].shape != expected_input_shapes_[i])
            {
                throw std::runtime_error(
                    "axflow::Postamble: input " + std::to_string(i) + " shape mismatch — got "
                    + shape_str(inputs[i].shape) + ", graph expects "
                    + shape_str(expected_input_shapes_[i]));
            }
        }

        // build ORT input tensors
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(inputs.size());

        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            std::vector<int64_t> shape64(inputs[i].shape.begin(), inputs[i].shape.end());
            ort_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                const_cast<float*>(inputs[i].data.data()),
                inputs[i].data.size(),
                shape64.data(),
                shape64.size()
            ));
        }

        // raw c-string arrays for the Run() call
        std::vector<const char*> input_name_ptrs;
        std::vector<const char*> output_name_ptrs;
        input_name_ptrs.reserve(input_names_.size());
        output_name_ptrs.reserve(output_names_.size());
        for (const auto& n : input_names_) input_name_ptrs.push_back(n.c_str());
        for (const auto& n : output_names_) output_name_ptrs.push_back(n.c_str());

        auto ort_outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(), ort_inputs.data(), ort_inputs.size(),
            output_name_ptrs.data(), output_name_ptrs.size()
        );

        // copy outputs into Tensor
        outputs_.clear();
        outputs_.reserve(ort_outputs.size());

        for (std::size_t i = 0; i < ort_outputs.size(); ++i)
        {
            auto& v = ort_outputs[i];
            auto info = v.GetTensorTypeAndShapeInfo();

            Tensor t;
            t.name = output_names_[i];

            auto raw_shape = info.GetShape();
            t.shape.reserve(raw_shape.size());
            for (auto d : raw_shape) t.shape.push_back(static_cast<int>(d));

            const std::size_t n_elem = info.GetElementCount();
            t.data.resize(n_elem);
            const float* src = v.GetTensorData<float>();
            std::copy(src, src + n_elem, t.data.begin());

            outputs_.push_back(std::move(t));
        }
        return outputs_;
    }

    int Postamble::find_output_index(const std::string& name) const
    {
        for (std::size_t i = 0; i < output_names_.size(); ++i)
        {
            if (output_names_[i] == name) return static_cast<int>(i);
        }
        throw std::runtime_error("axflow::Postamble: no output named '" + name + "'");
    }

    const Tensor& Postamble::get_output(const std::string& name) const
    {
        return outputs_.at(find_output_index(name));
    }

    const Tensor& Postamble::get_output(int index) const
    {
        return outputs_.at(index);
    }

    const std::string& Postamble::input_name(int i) const { return input_names_.at(i); }
    const std::string& Postamble::output_name(int i) const { return output_names_.at(i); }
} // namespace axflow
