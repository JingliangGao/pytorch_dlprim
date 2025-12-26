#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& addcdiv_out(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const Scalar& value,
    at::Tensor& out)
{

    at::Tensor self_c = self.contiguous();
    at::Tensor out_c = out.contiguous();
    at::Tensor tensor1_c = tensor1.contiguous();
    at::Tensor tensor2_c = tensor2.contiguous();

    dlprim::Tensor x0 = todp(self_c);
    dlprim::Tensor x1 = todp(tensor1_c);
    dlprim::Tensor x2 = todp(tensor2_c);
    dlprim::Tensor y0 = todp(out_c);
    float w0 = value.toDouble();
    dlprim::core::pointwise_operation_broadcast(
        {x0, x1, x2}, {y0}, {w0}, "y0 = x0 + w0 * (x1/x2);", getExecutionContext(self));

    if (!out.is_contiguous())
        out.copy_(out_c);

    sync_if_needed(self.device());
    return out;
}

} /* namespace op_plugin */
} /* namespace at_torch */
