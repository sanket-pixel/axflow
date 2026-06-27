#include "axflow/postamble/postamble.h"

#include "axflow/config/onnx_inference_config.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace axflow {
    namespace {
        std::string shape_str(const std::vector<int> &shape) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                out << shape[i];
                if (i + 1 < shape.size()) out << ",";
            }
            out << "]";
            return out.str();
        }
    } // anonymous namespace

    Postamble::Postamble(const PostambleConfig &config, const std::string &model_dir) {
        auto graph_full_path = std::filesystem::path(model_dir) / config.graph_path;
        if (!std::filesystem::exists(graph_full_path)) {
            throw std::runtime_error(
                "axflow::Postamble: graph not found at " + graph_full_path.string());
        }

        OnnxInferenceConfig ort_config;
        ort_config.model_path = graph_full_path.string();
        ort_config.intra_op_num_threads = config.intra_op_num_threads;

        engine_ = std::make_unique<OnnxInference>(ort_config);
    }

    Postamble::~Postamble() = default;

    std::vector<Tensor> &Postamble::run(const std::vector<Tensor> &aipu_outputs) {
        const int expected = engine_->input_count();
        const int got = static_cast<int>(aipu_outputs.size());

        if (got != expected) {
            throw std::runtime_error(
                "axflow::Postamble: expected " + std::to_string(expected)
                + " inputs, got " + std::to_string(got));
        }

        // copy each AIPU output into the matching ORT input slot.
        // shape-check first — fail loud if Voyager output order ever drifts.
        for (int i = 0; i < expected; ++i) {
            Tensor &ort_input = engine_->get_input_tensor(i);
            const Tensor &aipu = aipu_outputs[i];

            if (aipu.shape != ort_input.shape) {
                throw std::runtime_error(
                    "axflow::Postamble: input " + std::to_string(i)
                    + " shape mismatch — AIPU produced " + shape_str(aipu.shape)
                    + ", ORT graph expects " + shape_str(ort_input.shape));
            }

            if (aipu.data.size() != ort_input.data.size()) {
                throw std::runtime_error(
                    "axflow::Postamble: input " + std::to_string(i)
                    + " size mismatch");
            }

            std::copy(aipu.data.begin(), aipu.data.end(), ort_input.data.begin());
        }

        return engine_->run();
    }
} // namespace axflow
