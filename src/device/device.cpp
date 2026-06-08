#include "axflow/device/device.h"
#include "device/device_internal.h"

#include <stdexcept>
#include <string>

namespace axflow {

Device::Device() : impl_(std::make_unique<Impl>()) {
    impl_->ctx = axr_create_context();
    if (!impl_->ctx) {
        throw std::runtime_error("axflow::Device: failed to create context");
    }

    impl_->connection = axr_device_connect(impl_->ctx, nullptr, 1, nullptr);
    if (!impl_->connection) {
        std::string err = axr_last_error_string(AXR_OBJECT(impl_->ctx));
        throw std::runtime_error("axflow::Device: connect failed: " + err);
    }
}

Device::~Device() {
    if (impl_ && impl_->connection) {
        axr_destroy(reinterpret_cast<const axrObject*>(impl_->connection));
    }
    if (impl_ && impl_->ctx) {
        axr_destroy(reinterpret_cast<const axrObject*>(impl_->ctx));
    }
}

Device::Device(Device&&) noexcept            = default;
Device& Device::operator=(Device&&) noexcept = default;

bool Device::is_connected() const {
    return impl_ && impl_->connection != nullptr;
}

Device::Impl* internal_impl(Device& d) {
    return d.impl_.get();
}

} // namespace axflow