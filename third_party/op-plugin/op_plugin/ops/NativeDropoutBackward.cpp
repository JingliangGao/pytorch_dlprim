#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale)
    {

        at::Tensor grad_output_c=grad_output.contiguous();
        at::Tensor mask_c=mask.contiguous();
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor m = todp(mask_c);
        at::Tensor res =  new_tensor_as(dy.shape(),grad_output);
        dlprim::Tensor dx = todp(res);
        dlprim::core::pointwise_operation({dy,m},{dx},{scale},
                "y0 = x0*x1*w0;",
                getExecutionContext(grad_output));
        sync_if_needed(grad_output.device());
        return res;
    }


  }  /* namespace op_plugin */
}  /* namespace at_torch */
