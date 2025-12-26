#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

Scalar _local_scalar_dense(const at::Tensor& self)
{

    TORCH_CHECK(self.numel() == 1);
    dlprim::Tensor x = todp(self);
    union
    {
        float f;
        double d;
        int8_t i8;
        uint8_t u8;
        int16_t i16;
        uint16_t u16;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        uint64_t u64;
        char data[16];
    } data;
    x.to_host(getExecutionContext(self), data.data);
    switch (x.dtype())
    {
    case dlprim::float_data:
        return data.f;
    case dlprim::double_data:
        return data.d;
    case dlprim::int8_data:
        return data.i8;
    case dlprim::uint8_data:
        return data.u8;
    case dlprim::int16_data:
        return data.i16;
    case dlprim::uint16_data:
        return data.u16;
    case dlprim::int32_data:
        return (int64_t)data.i32;
    case dlprim::uint32_data:
        return (int64_t)data.u32;
    case dlprim::int64_data:
        return (int64_t)data.i64;
    case dlprim::uint64_data:
        return (int64_t)data.u64;
    default:
        TORCH_CHECK(!"Not implemented dtype", "Not implemented data type");
    }
}

} /* namespace op_plugin */
} /* namespace at_torch */
