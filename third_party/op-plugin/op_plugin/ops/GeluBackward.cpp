#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& gelu_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate,
    at::Tensor& grad_input)
{

    at::Tensor grad_output_c = grad_output.contiguous(), grad_input_c = grad_input.contiguous(),
               self_c = self.contiguous();

    dlprim::Tensor dX = todp(grad_input_c);
    dlprim::Tensor dY = todp(grad_output_c);
    dlprim::Tensor X = todp(self_c);

    auto q = getExecutionContext(self);
    TORCH_CHECK(approximate == "none" || approximate == "tanh", "Unsupported variant")

    char const* eq;
    // 1.128379167095512558561 = 2/ sqrt(pi)
    // 0.7071067811865475 = 1/sqrt(2)

    if (approximate == "tanh")
        eq = R"xxx(
                dtype alpha = 1.128379167095512558561f * 0.7071067811865475f;
                dtype koeff = 0.044715f;
                dtype beta  = alpha * koeff * 3.0f;
                dtype Y = tanh(alpha * fma(koeff,x0*x0*x0,x0));
                y0 = 0.5f * x1 * fma(
                    fma(-x0,Y * Y, x0),
                    fma(beta,x0*x0,alpha),
                    1 + Y
                );
            )xxx";
    else
        eq = R"xxx(
                dtype alpha = 1.128379167095512558561f * 0.7071067811865475f * 0.5f;
                dtype cdf = 0.5f * (1.0f + erf(x0 * 0.7071067811865475f));
                y0 = x1 * fma(
                    alpha * x0,
                    exp(-0.5f * x0*x0),
                    cdf);
            )xxx";

    dlprim::core::pointwise_operation({X, dY}, {dX}, {}, eq, q);

    if (!grad_input.is_contiguous())
        grad_input.copy_(grad_input_c);

    sync_if_needed(self.device());
    return grad_input;
}

} /* namespace op_plugin */
} /* namespace at_torch */
