#pragma once

// internal — NOT part of public api.
// used by model.cpp to get raw axruntime handles from a Device.

#include "axflow/device/device.h"
#include "axruntime/axruntime.hpp"

namespace axflow {

class Device::Impl {
public:
    axrContext*    ctx        = nullptr;
    axrConnection* connection = nullptr;
};

// defined in device.cpp
Device::Impl* internal_impl(Device& d);

} // namespace axflow