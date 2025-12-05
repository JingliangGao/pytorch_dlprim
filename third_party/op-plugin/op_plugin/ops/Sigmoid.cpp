#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & sigmoid_out(const at::Tensor & self, at::Tensor & out)
    {

        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();

        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::sigmoid,getExecutionContext(self));

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }

    at::Tensor sigmoid(const at::Tensor & self)
    {

        at::Tensor self_c = self.contiguous();

        dlprim::Tensor x=todp(self_c);
        at::Tensor out = new_tensor_as(x.shape(),self);

        dlprim::Tensor y=todp(out);

        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::sigmoid,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    at::Tensor & sigmoid_(at::Tensor & self)
    {

        at::Tensor self_c = self.contiguous();

        dlprim::Tensor X=todp(self_c);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::sigmoid,getExecutionContext(self));

        if(!self.is_contiguous())
          self.copy_(self_c);

        sync_if_needed(self.device());
        return self;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
