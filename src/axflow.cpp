#include "axflow/axflow.h"
#include "axflow/inference/generate_inference.h"

#include <stdexcept>

namespace axflow {
    AxFlow::AxFlow(Device &device, const std::string &config_path)
        : config_(AxflowConfig::from_yaml(config_path)),
          preprocessor_(config_.preprocessing) {
        init_engines(device);
    }

    AxFlow::AxFlow(Device &device, const AxflowConfig &config)
        : config_(config),
          preprocessor_(config_.preprocessing) {
        init_engines(device);
    }

    void AxFlow::init_engines(Device &device) {
        // factory builds the polymorphic engine (aipu or onnx).
        // engine allocates and locks its own physical memory at this point.
        engine_ = generate_inference(device, config_);

        if (config_.postamble.enabled) {
            postamble_ = std::make_unique<Postamble>(
                config_.postamble, config_.axruntime_inference.model_dir);
        }
    }

    void AxFlow::preprocess(const cv::Mat &image) {
        if (!engine_) {
            throw std::runtime_error("axflow::AxFlow: engine not initialized for preprocessing");
        }
        // preprocessor writes directly into the engine's mapped input memory.
        preprocessor_.run(image, engine_->get_input_tensor(0));
    }

    std::vector<Tensor> AxFlow::inference() {
        if (!engine_) {
            throw std::runtime_error("axflow::AxFlow: inference engine not initialized");
        }
        raw_outputs_ = engine_->run();

        if (postamble_) {
            return postamble_->run(raw_outputs_);
        }
        return raw_outputs_;
    }

    const std::vector<Tensor> &AxFlow::raw_aipu_outputs() const {
        return raw_outputs_;
    }
} // namespace axflow
