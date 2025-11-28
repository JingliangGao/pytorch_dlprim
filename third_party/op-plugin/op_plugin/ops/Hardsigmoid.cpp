#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & hardsigmoid_out(const at::Tensor & self, at::Tensor & out)
    {
        GUARD;

        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();

        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::pointwise_operation({x},{y},{},"y0 = x0 <= -3.0f ? 0 : (x0>=3.0f ? 1.0f : x0/6.0f + 0.5f);",getExecutionContext(self));

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;

    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
