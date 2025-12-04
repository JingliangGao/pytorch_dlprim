#include "HooksInterface.h"

namespace at_torch {

    bool KPUHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const
    {
        return CLContextManager::is_ready(device_index);
    }

    at::PrivateUse1HooksInterface* get_kpu_hooks()
    {
        static at::PrivateUse1HooksInterface* kpu_hooks;
        static c10::once_flag once;
        c10::call_once(once, [] {
            kpu_hooks = new KPUHooksInterface();
        });
        return kpu_hooks;
    }

} // namespace at_torch
