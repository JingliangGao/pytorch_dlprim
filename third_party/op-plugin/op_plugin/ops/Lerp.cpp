#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const Scalar & weight, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor end_c = end.contiguous();
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor x1=todp(end_c);
        dlprim::Tensor y0 = todp(out);
        float w = weight.toDouble();

        dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{w},
                                      "y0 = x0 + w0 * (x1 - x0 );",
                                      getExecutionContext(self));
        sync_if_needed(self.device());
        return out;

    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
