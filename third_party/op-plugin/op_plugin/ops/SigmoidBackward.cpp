#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& sigmoid_backward_out(
    const at::Tensor& grad_output, const at::Tensor& output, at::Tensor& grad_input)
{

    at::Tensor output_c = output.contiguous(), grad_output_c = grad_output.contiguous(),
               grad_input_c = grad_input.contiguous();

    dlprim::Tensor y = todp(output_c);
    dlprim::Tensor dy = todp(grad_output_c);

    dlprim::Tensor dx = todp(grad_input_c);
    dlprim::core::activation_backward(
        dx, dy, y, dlprim::StandardActivations::sigmoid, 0, getExecutionContext(grad_output));

    if (!grad_input.is_contiguous())
        grad_input.copy_(grad_input_c);

    sync_if_needed(grad_output.device());
    return grad_input;
}

} /* namespace op_plugin */
} /* namespace at_torch */
