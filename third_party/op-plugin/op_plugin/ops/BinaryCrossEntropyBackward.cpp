#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & binary_cross_entropy_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight in binar_cross_entroy isn't supported");
        at::Tensor self_c = self.contiguous();
        at::Tensor target_c = target.contiguous();
        at::Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(target_c);
        dlprim::Tensor dloss = todp(grad_output_c);
        double scale = 1;
        if(reduction == 1) // mean
            scale = 1.0/x.shape().total_size();
        dlprim::Tensor dx = todp(grad_input);

        // -w (y - x) / (x - x^2)
        dlprim::core::pointwise_operation_broadcast({x,y,dloss},{dx},{scale},
                "y0 = -(x1 - x0) / max(1e-12f,x0 - x0*x0) * x2 * w0;",
                getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;

    }

    at::Tensor binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor input_grad = new_tensor_as(todp(self_c).shape(),self_c);
        binary_cross_entropy_backward_out(grad_output,self_c,target,weight,reduction,input_grad);
        return input_grad;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
