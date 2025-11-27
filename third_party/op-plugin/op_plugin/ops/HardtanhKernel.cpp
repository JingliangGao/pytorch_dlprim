#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor hardtanh(at::Tensor const &self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        at::Tensor out = new_tensor_as(X.shape(),self);
        dlprim::Tensor Y(todp(out));
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X},{Y},{w0,w1},"y0=max(w0,min(w1,x0));",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

 
    at::Tensor & hardtanh_(at::Tensor & self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X=todp(self_c);
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X},{X},{w0,w1},"y0=max(w0,min(w1,x0));",getExecutionContext(self));
        if(!self.is_contiguous())
            self.copy_(self_c);
        sync_if_needed(self.device());
        return self;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */