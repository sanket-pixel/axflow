#pragma once

#include "axflow/model/model.h"
#include "axruntime/axruntime.hpp"

#include <cstdint>
#include <vector>

namespace axflow {

class Model::Impl {
public:
    // axruntime handles (not owned: ctx, connection)
    axrContext*       ctx        = nullptr;
    axrConnection*    connection = nullptr;

    // owned
    axrModel*         model      = nullptr;
    axrModelInstance* instance   = nullptr;

    // tensor info — cached at load time
    std::vector<axrTensorInfo>       input_infos;
    std::vector<axrTensorInfo>       output_infos;

    // pre-allocated int8 buffers
    std::vector<std::vector<int8_t>> input_bufs;
    std::vector<std::vector<int8_t>> output_bufs;
};

} // namespace axflow