#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & hardswish_(at::Tensor & self)
    {
        GUARD;
        
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        dlprim::core::pointwise_operation({x},{x},{},"y0 = x0 <= -3.0f ? 0 : (x0>=3.0f ? x0 : x0*(x0+3.0f)/6.0f);",getExecutionContext(self));
        
        if (!self.is_contiguous())
            self.copy_(self_c);
        
        sync_if_needed(self.device());
        return self;
    }
    

    }  /* namespace op_plugin */
}  /* namespace at_torch */