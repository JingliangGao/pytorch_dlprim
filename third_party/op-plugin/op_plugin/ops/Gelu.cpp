#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & gelu_out(const at::Tensor & self, c10::string_view approximate, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        dlprim::Tensor X = todp(self_c);
        auto q = getExecutionContext(self);
        TORCH_CHECK(approximate == "none" || approximate == "tanh","Unsupported variant")
        if(approximate == "tanh")
            dlprim::core::pointwise_operation({X},{Y},{},"y0 = 0.5f * x0 * (1.0f + tanh(0.7978845608028654f * x0 * (1.0f + 0.044715f * x0 * x0)));",q); // 0.7978845608028654 = sqrt(2/pi)
        else {
            dlprim::core::pointwise_operation({X},{Y},{},"y0 = x0 * (1.0f + erf(x0 * 0.7071067811865475f  )) / 2.0f;",q); // 0.7071067811865475 = 1/sqrt(2)
        }
            
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */