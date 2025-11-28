#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor mse_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction)
    {
        GUARD;
        at::Tensor grad_output_c = grad_output.contiguous();
        at::Tensor self_c = self.contiguous();
        at::Tensor target_c = target.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor lbl = todp(target_c);
        at::Tensor result = new_tensor_as(x.shape(),self_c);
        dlprim::Tensor dx = todp(result);
        double scale = reduction == 1 ? (1.0f/x.shape().total_size()) : 1.0;
        dlprim::core::pointwise_operation_broadcast({dy,x,lbl},{dx},{scale},
            "y0 = 2*(x1 -x2) * x0 * w0;",getExecutionContext(self.device()));
        sync_if_needed(self.device());
        return result;
    }


}  /* namespace op_plugin */
}  /* namespace at_torch */
