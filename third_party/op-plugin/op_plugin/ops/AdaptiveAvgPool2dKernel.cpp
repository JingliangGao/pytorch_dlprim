#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor _adaptive_avg_pool2d(const at::Tensor & self, IntArrayRef output_size) 
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        int h=X.shape()[2];
        int w=X.shape()[3];
        TORCH_CHECK((output_size[0]==1 && output_size[1]==1) || (output_size[0]==h && output_size[1]==w),"Only global pooling or no-pooling supported");
        at::Tensor result;
        if(output_size[0]==1 && output_size[1] == 1) {
            result = new_tensor_as(dlprim::Shape(X.shape()[0],X.shape()[1],1,1),self);
            dlprim::Tensor Y = todp(result);

            dlprim::ExecutionContext q = getExecutionContext(self);
            dlprim::Context ctx(q);
            auto pool = dlprim::core::Pooling2DForward::create_global_avg_pooling(ctx,X.shape(),todp(self.dtype()));
            pool->enqueue(X,Y,q);
        }
        else {
            result = new_tensor_as(X.shape(),self);
            dlprim::Tensor Y = todp(result);
            dlprim::core::pointwise_operation({X},{Y},{},"y0=x0;",getExecutionContext(self));
        }
        sync_if_needed(self.device());
        return result;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */