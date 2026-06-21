#include "axflow/device/device.h"

#include <stdexcept>
#include <string>

namespace axflow {
  Device::Device() {
    context_ = axr_create_context();
    if (!context_) return; // mark as disconnected, don't throw
    connection_ = axr_device_connect(context_, nullptr, 1, nullptr);
    if (!connection_) {
      axr_destroy(reinterpret_cast<const axrObject *>(context_));
      context_ = nullptr;
    }
  }

  Device::~Device() {
    if (connection_)
      axr_destroy(reinterpret_cast<const axrObject *>(connection_));
    if (context_)
      axr_destroy(reinterpret_cast<const axrObject *>(context_));
  }
} // namespace axflow
