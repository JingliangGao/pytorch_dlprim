#pragma once
#include <torch/version.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include "CLTensor.h"
#include "HooksInterface.h"


namespace at_torch {

class KPUGuardImpl final: public c10::impl::DeviceGuardImplInterface {
public:
    KPUGuardImpl() { }
    explicit KPUGuardImpl(c10::DeviceType t);

    /* device management */
    c10::DeviceType type() const override;
    c10::Device exchangeDevice(c10::Device d) const override;
    c10::Device getDevice() const override;
    void setDevice(c10::Device d) const override;
    void uncheckedSetDevice(c10::Device d) const noexcept override;
    c10::DeviceIndex deviceCount() const noexcept override;

    /* stream management */
    c10::Stream getStream(c10::Device d) const noexcept override;
    c10::Stream getDefaultStream(c10::Device d) const override;
    c10::Stream exchangeStream(c10::Stream) const noexcept override;
    bool queryStream(const c10::Stream& ) const override;
    void synchronizeStream(const c10::Stream& stream) const override;


private:

    static thread_local c10::Device current_device_;
    static thread_local c10::Stream current_stream_;
};

}
