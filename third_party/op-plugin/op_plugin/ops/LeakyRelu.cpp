#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & leaky_relu_out(const at::Tensor & self, const Scalar & negative_slope, at::Tensor & out)
    {
        GUARD;
        double slope = negative_slope.to<double>();
        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();

        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::pointwise_operation({x},{y},{slope},"y0 = x0 > 0 ? x0 : w0 * x0;",getExecutionContext(self));

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }

    at::Tensor & leaky_relu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const Scalar & negative_slope, bool /*self_is_result*/, at::Tensor & grad_input)
    {
        GUARD;
        double slope = negative_slope.to<double>();
        at::Tensor self_c = self.contiguous(),
               grad_input_c  = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous();

        dlprim::Tensor y=todp(self_c);
        dlprim::Tensor dy=todp(grad_output_c);
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::core::pointwise_operation({y,dy},{dx},{slope},"y0 = x0 > 0 ? x1 : w0 * x1;",getExecutionContext(self));

        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);

        sync_if_needed(self.device());
        return grad_input;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
