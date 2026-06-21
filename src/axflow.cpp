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
        // 1. Factory generates the polymorphic engine.
        // The engine fully allocates and locks its own physical memory here.
        engine_ = generate_inference(device, config_);

        // 2. Initialize Postamble if enabled
        if (config_.postamble.enabled) {
            postamble_ = std::make_unique<Postamble>(config_.postamble, config_.inference.model_dir);
        }
    }

    void AxFlow::preprocess(const cv::Mat &image) {
        if (!engine_) {
            throw std::runtime_error("axflow::AxFlow: engine not initialized for preprocessing");
        }
        // Ask the engine for the direct reference to its mapped memory.
        // The Preprocessor writes natively into it.
        preprocessor_.run(image, engine_->get_input_tensor(0));
    }

    std::vector<Tensor> &AxFlow::inference() {
        if (!engine_) {
            throw std::runtime_error("axflow::AxFlow: inference engine not initialized");
        }

        // Execute. Zero copies, zero allocations.
        return engine_->run();
    }

    std::vector<Tensor> &AxFlow::postamble() {
        if (!postamble_) {
            throw std::runtime_error("axflow::AxFlow: postamble called but not enabled in config");
        }

        // We pass the output memory from the inference engine directly into the postamble graph
        postamble_outputs_ = postamble_->run(engine_->run());
        return postamble_outputs_;
    }
} // namespace axflow
