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

        const std::size_t expected = postamble_->num_inputs();
        if (inference_outputs_.size() > expected)
        {
            std::vector<Tensor> sliced(
                inference_outputs_.begin(),
                inference_outputs_.begin() + expected
            );
            return postamble_->run(sliced);
        }

        return postamble_->run(inference_outputs_);
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
