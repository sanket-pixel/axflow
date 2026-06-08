#pragma once

#include <memory>

namespace axflow {

// owns axruntime context + connection — one per process
// models borrow this via reference
class Device {
public:
    Device();
    ~Device();

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    bool is_connected() const;

    // pimpl — full type only visible inside src/
    class Impl;

private:
    std::unique_ptr<Impl> impl_;

    // src/device/device_internal.h exposes this to Model::Impl
    friend Impl* internal_impl(Device&);
};

} // namespace axflow