#pragma once

#include <axflow/config/config_base.h>

#include <string>
#include <yaml-cpp/yaml.h>

namespace axflow {
    // inference block — points at compiled chip artifacts and runtime knobs
    class AxruntimeInferenceConfig : public ConfigBase {
    public:
        std::string model_dir{"model"};
        int num_cores = 1;

    protected:
        void parse(const YAML::Node &node) override;
    };
} // namespace axflow
