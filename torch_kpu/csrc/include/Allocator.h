#include "CLTensor.h"
#include "GuardImpl.h"
#include "Utils.h"
#include <torch/version.h>

namespace at_torch
{

class KPUAllocator : public c10::Allocator
{

  public:
    c10::DataPtr allocate(size_t nbytes) override;
    void copy_data(void* dest, const void* src, std::size_t count) const;
};

} // namespace at_torch
