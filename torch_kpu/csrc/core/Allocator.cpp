
#include "Allocator.h"
#include "ProfilerInterface.h"

namespace at_torch
{

at_torch::KPUGuardImpl kpu_guard_impl_instance;

c10::DataPtr KPUAllocator::allocate(size_t nbytes)
{
    c10::Device device = kpu_guard_impl_instance.getDevice();
    ;
    return CLContextManager::allocate(device, nbytes);
}

void KPUAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    GUARD;
    c10::Device device = kpu_guard_impl_instance.getDevice();
    ;
    cl::Buffer buf_dst((cl_mem)dest, true);
    cl::Buffer buf_src((cl_mem)src, true);
    auto q = getExecutionContext(device);
    q.queue().enqueueCopyBuffer(buf_src, buf_dst, 0, 0, count, q.events(), q.event("copy_data"));
    sync_if_needed(device);
}

// Register allocator
KPUAllocator kpu_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &kpu_allocator);

} // namespace at_torch
