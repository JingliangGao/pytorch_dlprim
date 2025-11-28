#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor relu(const at::Tensor & self)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor x = todp(self_c);
        at::Tensor out = new_tensor_as(x.shape(), self);
        dlprim::Tensor y = todp(out);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::relu, getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }


    at::Tensor & relu_(at::Tensor & self)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::ExecutionContext q = getExecutionContext(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::relu,q);

        if (!self.is_contiguous())
            self.copy_(self_c);

        sync_if_needed(self.device());
        return self;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
