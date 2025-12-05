#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self)
    {

        at::Tensor self_c = self.contiguous();
        at::Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor dy = todp(grad_output_c);
        at::Tensor result;
        TORCH_CHECK((dy.shape()[2]==1 && dy.shape()[3]==1) || (dy.shape() == X.shape()),"Only global pooling or no-pooling supported");
        if(dy.shape()[2]==1 && dy.shape()[3]==1) {
            result = new_tensor_as(X.shape(),self);
            dlprim::Tensor dx = todp(result);

            dlprim::ExecutionContext q = getExecutionContext(self);
            dlprim::Context ctx(q);
            auto pool = dlprim::core::AvgPooling2DBackward::create_global(ctx,X.shape(),todp(self.dtype()));
            pool->enqueue(dx,dy,0,q);
        }
        else {
            result = new_tensor_as(dy.shape(),self);
            dlprim::Tensor dx = todp(result);
            dlprim::core::pointwise_operation({dy},{dx},{},"y0=x0;",getExecutionContext(self));
        }
        sync_if_needed(self.device());
        return result;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
