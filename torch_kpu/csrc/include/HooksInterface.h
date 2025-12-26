#pragma once
#include "CLTensor.h"
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at_torch
{

struct TORCH_API KPUHooksInterface : public at::PrivateUse1HooksInterface
{
    virtual ~KPUHooksInterface() = default;
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
};

// register to PrivateUse1HooksInterface
at::PrivateUse1HooksInterface* get_kpu_hooks();

} // namespace at_torch
