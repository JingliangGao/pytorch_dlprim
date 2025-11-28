#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & tanh_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input)
    {
        GUARD;
        at::Tensor grad_input_c  = grad_input.contiguous();
        at::Tensor grad_output_c = grad_output.contiguous();
        at::Tensor output_c      = output.contiguous();

        dlprim::Tensor dY=todp(grad_output_c);
        dlprim::Tensor Y=todp(output_c);
        dlprim::Tensor dX=todp(grad_input_c);
        dlprim::core::activation_backward(dX,dY,Y,dlprim::StandardActivations::tanh,0.0,getExecutionContext(grad_output));

        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);

        sync_if_needed(grad_output.device());
        return grad_input;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
