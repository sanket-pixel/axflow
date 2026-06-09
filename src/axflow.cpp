#include "axflow/axflow.h"

namespace axflow
{
    AxFlow::AxFlow(Device& device, const std::string& config_path)
        : AxFlow(device, AxflowConfig::from_yaml(config_path))
    {
    }

    AxFlow::AxFlow(Device& device, const AxflowConfig& config)
        : config_(config),
          preprocessor_(config_.preprocessing),
          inference_(device, config_.inference),
          postprocessor_(config_.postprocessing)
    {
    }

    void AxFlow::preprocess(const cv::Mat& image)
    {
        // single-input case for now — first input buffer
        auto input = inference_.get_input(0);
        preprocessor_.run(image, input);
    }

    std::vector<Tensor> AxFlow::inference()
    {
        inference_outputs_ = inference_.run();
        return inference_outputs_;
    }

    std::vector<Tensor> AxFlow::postprocess()
    {
        return postprocessor_.run(inference_outputs_);
    }
} // namespace axflow
