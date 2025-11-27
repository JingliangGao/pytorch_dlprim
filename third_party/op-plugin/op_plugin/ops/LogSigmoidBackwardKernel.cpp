#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & log_sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input)
    {
        GUARD;
        at::Tensor grad_output_c = grad_output.contiguous();
        at::Tensor self_c = self.contiguous();
        at::Tensor buffer_c = buffer.contiguous();
        at::Tensor grad_input_c = grad_input.contiguous();

        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor buf = todp(buffer_c);
        dlprim::Tensor dx = todp(grad_input_c);

        dlprim::core::pointwise_operation({x,buf,dy},{dx},{},
                    R"xxx(
                    int is_negative = x0 < 0;
                    dtype maxd = is_negative ? 1.0f: 0.0f;
                    dtype s = is_negative ? 1.0f: -1.0f;
                    y0 = (maxd - s * (x1 / ((dtype)(1) + x1))) * x2;
                    )xxx",
                    getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);;

        sync_if_needed(self.device());
        return grad_input;
    }


    at::Tensor log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer)
    {
        GUARD;
        at::Tensor grad_input = at::empty_like(grad_output);
        log_sigmoid_backward_out(grad_output,self,buffer,grad_input);
        return grad_input;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */