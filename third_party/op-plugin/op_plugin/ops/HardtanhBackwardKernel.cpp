#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor hardtanh_backward(const at::Tensor & grad_output, const at::Tensor & self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X  = todp(self_c);
        dlprim::Tensor dY = todp(grad_output);
        at::Tensor result = new_tensor_as(X.shape(),self);
        dlprim::Tensor dX = todp(result);
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X,dY},{dX},{w0,w1},"y0 = (w0 <= x0 && x0 <= w1) ? x1 : 0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return result;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */