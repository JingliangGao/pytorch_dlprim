#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset)
{

    at::Tensor result = at::alias(self);
    result.getIntrusivePtr()->set_sizes_and_strides(size, stride);
    if (storage_offset)
        result.getIntrusivePtr()->set_storage_offset(*storage_offset);
    return result;
}

at::Tensor _reshape_alias(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride)
{

    at::Tensor data = at::alias(self);
    data.getIntrusivePtr()->set_sizes_and_strides(size, stride);
    return data;
}

} /* namespace op_plugin */
} /* namespace at_torch */
