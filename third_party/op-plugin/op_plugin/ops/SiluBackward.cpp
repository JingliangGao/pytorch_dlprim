#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor&
silu_backward_out(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& grad_input)
{

    at::Tensor grad_output_c = grad_output.contiguous(), grad_input_c = grad_input.contiguous(),
               self_c = self.contiguous();

    dlprim::Tensor x = todp(self_c);
    dlprim::Tensor dy = todp(grad_output_c);
    dlprim::Tensor dx = todp(grad_input);
    dlprim::core::pointwise_operation(
        {x, dy},
        {dx},
        {},
        R"xxx(
                y0 = 1.0f / (1.0f + exp(-x0));
                y0 = x1 * y0 * ( 1.0f + x0 * (1.0f - y0));
            )xxx",
        getExecutionContext(self));

    if (!grad_input.is_contiguous())
        grad_input.copy_(grad_input_c);

    sync_if_needed(self.device());
    return grad_input;
}

} /* namespace op_plugin */
} /* namespace at_torch */
