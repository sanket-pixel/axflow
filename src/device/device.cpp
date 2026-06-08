#include "axflow/device/device.h"

#include <stdexcept>
#include <string>

namespace axflow {

Device::Device() {
  context_ = axr_create_context();
  if (!context_) {
    throw std::runtime_error("axflow::Device: failed to create context");
  }

  connection_ = axr_device_connect(context_, nullptr, 1, nullptr);
  if (!connection_) {
    std::string error = axr_last_error_string(AXR_OBJECT(context_));
    throw std::runtime_error("axflow::Device: connect failed: " + error);
  }
}

Device::~Device() {
  if (connection_)
    axr_destroy(reinterpret_cast<const axrObject *>(connection_));
  if (context_)
    axr_destroy(reinterpret_cast<const axrObject *>(context_));
}

} // namespace axflow