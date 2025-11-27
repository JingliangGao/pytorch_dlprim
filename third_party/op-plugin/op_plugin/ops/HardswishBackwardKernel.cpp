#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & hardsigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input)
    {
        GUARD;
        at::Tensor self_c = self.contiguous(),
               grad_input_c = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous();
        
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::Tensor dy=todp(grad_output_c);

        dlprim::core::pointwise_operation({x,dy},{dx},{},"y0 = (-3 < x0 && x0 < 3) ? x1 / 6 : 0;",getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(self.device());
        return grad_input;
    }
    

    at::Tensor hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self)
    {
        GUARD;
        at::Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor dy=todp(grad_output_c);
        
        at::Tensor out = new_tensor_as(dy.shape(),grad_output);
        dlprim::Tensor dx=todp(out);
        
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor x =todp(self_c);
        dlprim::core::pointwise_operation({x,dy},{dx},{},
            R"xxx(
                if (x0 < -3.0f) {
                    y0 = 0;
                } else if (x0 <= 3.0f) {
                    y0 =  x1 * ((x0 / 3.0f) + 0.5f);
                } else {
                    y0 = x1;
                }
            )xxx",
            getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }
    

    }  /* namespace op_plugin */
}  /* namespace at_torch */