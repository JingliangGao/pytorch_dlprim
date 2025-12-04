#include "GuardImpl.h"

namespace at_torch {

    thread_local c10::Device KPUGuardImpl::current_device_ = c10::Device(c10::DeviceType::PrivateUse1,0);
    thread_local c10::Stream KPUGuardImpl::current_stream_ = c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::PrivateUse1,0),0);

    /* check device type */
    KPUGuardImpl::KPUGuardImpl(c10::DeviceType t)
    {
        TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", t);
    }

    /* return device type */
    c10::DeviceType KPUGuardImpl::type() const
    {
        return c10::DeviceType::PrivateUse1;
    }

    /* exchange device */
    c10::Device KPUGuardImpl::exchangeDevice(c10::Device d) const
    {
        TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", d.type());
        c10::Device prev_device = current_device_;
        current_device_ = d;
        return prev_device;
    }

    /* get current device */
    c10::Device KPUGuardImpl::getDevice() const {
        return current_device_;
    }

    /* set current device */
    void KPUGuardImpl::setDevice(c10::Device d) const {
        TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", d.type());
        current_device_ = d;
    }

    /* set current device without checking */
    void KPUGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept  {
        current_device_ = d;
    }

    /* get numbers of device */
    c10::DeviceIndex KPUGuardImpl::deviceCount() const noexcept {
        try {
            return CLContextManager::count();
        }
        catch(...) {
            return 0;
        }
    }

    /* get current stream */
    c10::Stream KPUGuardImpl::getStream(c10::Device d) const noexcept {
        TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", d.type());
        return c10::Stream(c10::Stream::UNSAFE,d,0);
    }

    /* get default stream */
    c10::Stream KPUGuardImpl::getDefaultStream(c10::Device d) const {
        return getStream(d);
    }

    /* exchange stream */
    c10::Stream KPUGuardImpl::exchangeStream(c10::Stream s) const noexcept {
        TORCH_INTERNAL_ASSERT(s.device().type() == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", s.device().type());
        c10::Stream prev_stream = current_stream_;
        current_stream_ = s;
        return prev_stream;
    }

    /* query stream */
    bool KPUGuardImpl::queryStream(const c10::Stream& s) const {
        TORCH_INTERNAL_ASSERT(s.device().type() == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", s.device().type());
        return false;
    }

    /* synchronize stream */
    void KPUGuardImpl::synchronizeStream(const c10::Stream& stream) const {
        TORCH_INTERNAL_ASSERT(stream.device().type() == c10::DeviceType::PrivateUse1, "DeviceType must be KPU. Actual DeviceType is: ", stream.device().type());
        auto device = stream.device();
        CLContextManager::getCommandQueue(device.index()).finish();
    }


// Register Guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, KPUGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                                                             \
    int rename_privateuse1_backend()                                                                                   \
    {                                                                                                                  \
        c10::register_privateuse1_backend(#name);                                                                      \
        at::RegisterPrivateUse1HooksInterface(at_torch::get_kpu_hooks());                                              \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static const int _temp_##name = rename_privateuse1_backend();

// Register backend
REGISTER_PRIVATEUSE1_BACKEND(kpu)

}
