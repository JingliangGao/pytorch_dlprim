#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & threshold_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const Scalar & threshold, at::Tensor & grad_input)
    {
        GUARD;
        at::Tensor self_c = self.contiguous(),
               grad_input_c = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous();

        dlprim::Tensor dy=todp(grad_output_c);
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::Tensor Y=todp(self_c);
        float th = threshold.toDouble();
        dlprim::core::pointwise_operation({Y,dy},{dx},{th},"y0 = (x0 > w0) ? x1 : 0;",getExecutionContext(self));

        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);

        sync_if_needed(self.device());
        return grad_input;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
