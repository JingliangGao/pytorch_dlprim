#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor &zero_(at::Tensor &self)
    {
        GUARD;
        if(self.numel() == 0)
            return self;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor t(todp(self));
        dlprim::core::fill_tensor(t,0.0,getExecutionContext(self));
        if(!self.is_contiguous())
            self.copy_(self_c);
        return self;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
