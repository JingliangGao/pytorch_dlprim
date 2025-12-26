#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& avg_pool2d_out(
    const at::Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool /*ceil_mode*/,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& out)
{

    TORCH_CHECK(!divisor_override, "Divisor override is not implemented");
    // note ceil mode calculations are based on output size
    int ker[2] = {int(kernel_size[0]), int(kernel_size[1])};
    int pad[2] = {int(padding[0]), int(padding[1])};
    int strd[2];
    if (stride.empty())
    {
        strd[0] = ker[0];
        strd[1] = ker[1];
    }
    else
    {
        strd[0] = stride[0];
        strd[1] = stride[1];
    };
    at::Tensor self_c = self.contiguous();
    dlprim::Tensor X = todp(self_c);
    dlprim::Tensor Y = todp(out);
    dlprim::ExecutionContext q(getExecutionContext(self));
    dlprim::Context ctx(q);

    auto pool = dlprim::core::Pooling2DForward::create_avg_pooling(
        ctx, ker, pad, strd, count_include_pad, todp(self.dtype()));
    pool->enqueue(X, Y, q);
    sync_if_needed(self.device());
    return out;
}

} /* namespace op_plugin */
} /* namespace at_torch */
