#pragma once

#include "axruntime/axruntime.hpp"

namespace axflow {

// owns axruntime context + connection — one per process.
// models borrow this via reference.
class Device {
public:
  Device();
  ~Device();

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&) noexcept = default;
  Device &operator=(Device &&) noexcept = default;

  bool is_connected() const { return connection_ != nullptr; }

  // raw handles for use inside the library
  axrContext *context() const { return context_; }
  axrConnection *connection() const { return connection_; }

private:
  axrContext *context_ = nullptr;
  axrConnection *connection_ = nullptr;
};

} // namespace axflow