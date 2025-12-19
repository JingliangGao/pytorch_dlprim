#pragma once
#include <torch/version.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include "GuardImpl.h"

namespace at_torch {

    struct KPUGuard {
    //  No default constructor; see Note [Omitted default constructor from RAII]
    explicit KPUGuard() = delete;

    //  Set the current KPU device to the passed device index.
    explicit KPUGuard(c10::DeviceIndex device_index) : guard_(device_index) {}

    //  Sets the current KPU device to the passed device.  Errors if the passed
    //  device is not a KPU device.
    explicit KPUGuard(c10::Device device) : guard_(device) {}

    // Copy is not allowed
    KPUGuard(const KPUGuard &) = delete;
    KPUGuard &operator = (const KPUGuard &) = delete;

    // Move is not allowed (there is no uninitialized state)
    KPUGuard(KPUGuard &&other) = delete;
    KPUGuard &operator = (KPUGuard &&other) = delete;

    //  Sets the KPU device to the given device.  Errors if the given device
    //  is not a KPU device.
    void set_device(c10::Device device)
    {
        guard_.set_device(device);
    }

    //  Sets the KPU device to the given device.  Errors if the given device
    //  is not a KPU device.  (This method is provided for uniformity with
    //  DeviceGuard).
    void reset_device(c10::Device device)
    {
        guard_.reset_device(device);
    }

    //  Sets the KPU device to the given device index.
    void set_index(c10::DeviceIndex device_index)
    {
        guard_.set_index(device_index);
    }

    //  Returns the device that was set upon construction of the guard
    c10::Device original_device() const
    {
        return guard_.original_device();
    }

    //  Returns the last device that was set via `set_device`, if any, otherwise
    //  the device passed during construction.
    c10::Device current_device() const
    {
        return guard_.current_device();
    }

private:
    //  The guard for the current device.
    c10::impl::InlineDeviceGuard<at_torch::KPUGuardImpl> guard_;
};


}
