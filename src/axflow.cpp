#include "axflow/axflow.h"

#include <stdexcept>

namespace axflow
{
    AxFlow::AxFlow(Device& device, const std::string& config_path)
        : AxFlow(device, AxflowConfig::from_yaml(config_path))
    {
    }

    AxFlow::AxFlow(Device& device, const AxflowConfig& config)
        : config_(config),
          preprocessor_(config_.preprocessing),
          inference_(device, config_.inference)
    {
        if (config_.postamble.enabled)
        {
            postamble_ = std::make_unique<Postamble>(config_.postamble,
                                                     config_.inference.model_dir);
        }
    }

    void AxFlow::preprocess(const cv::Mat& image)
    {
        auto input = inference_.get_input(0);
        preprocessor_.run(image, input);
    }

    std::vector<Tensor> AxFlow::inference()
    {
        inference_outputs_ = inference_.run();
        return inference_outputs_;
    }

    std::vector<Tensor> AxFlow::postamble()
    {
        if (!postamble_)
        {
            throw std::runtime_error(
                "axflow::AxFlow::postamble: postamble is not enabled in config");
        }

        const std::size_t n_aipu = inference_outputs_.size();
        const std::size_t n_expected = postamble_->num_inputs();

        // slice first n_expected outputs for postamble
        std::vector<Tensor> postamble_inputs(
            inference_outputs_.begin(),
            inference_outputs_.begin() + std::min(n_expected, n_aipu)
        );

        // run postamble
        auto results = postamble_->run(postamble_inputs);

        // append any extra AIPU outputs that postamble didn't consume
        for (std::size_t i = n_expected; i < n_aipu; ++i)
            results.push_back(inference_outputs_[i]);

        return results;
    }

    Postamble& AxFlow::postamble_obj()
    {
        if (!postamble_)
        {
            throw std::runtime_error(
                "axflow::AxFlow::postamble_obj: postamble is not enabled in config");
        }
        return *postamble_;
    }
} // namespace axflow
