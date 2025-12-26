#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& div_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{

    at::Tensor self_c = self.contiguous();
    at::Tensor out_c = out.contiguous();

    dlprim::Tensor x0 = todp(self_c);
    dlprim::Tensor y0 = todp(out_c);
    double value = 0;
    if (is_cpu_scalar(other, value))
    {
        dlprim::core::pointwise_operation(
            {x0}, {y0}, {double(1.0 / value)}, "y0 = x0*w0;", getExecutionContext(self));
    }
    else
    {
        at::Tensor other_c = other.contiguous();
        dlprim::Tensor x1 = todp(other_c);
        dlprim::core::pointwise_operation_broadcast(
            {x0, x1}, {y0}, {}, "y0 = x0/x1;", getExecutionContext(self));
    }

    if (!out.is_contiguous())
        out.copy_(out_c);

    sync_if_needed(self.device());
    return out;
}

} /* namespace op_plugin */
} /* namespace at_torch */
