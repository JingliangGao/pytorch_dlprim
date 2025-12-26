#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& bitwise_not_out(const at::Tensor& self, at::Tensor& out)
{

    TORCH_CHECK(is_integral_type(self, true), "~ is valid for integer types");

    at::Tensor self_c = self.contiguous();
    at::Tensor out_c = out.contiguous();

    dlprim::Tensor x = todp(self_c);
    dlprim::Tensor y = todp(out_c);
    dlprim::core::pointwise_operation(
        {x},
        {y},
        {},
        (self.dtype() == c10::kBool ? "y0 = !x0;" : "y0 = ~x0;"),
        getExecutionContext(self));

    if (!out.is_contiguous())
        out.copy_(out_c);

    sync_if_needed(self.device());
    return out;
}

} /* namespace op_plugin */
} /* namespace at_torch */
