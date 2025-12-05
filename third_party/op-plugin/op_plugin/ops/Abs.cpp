#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & abs_out(const at::Tensor & self, at::Tensor & out)
    {


        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();

        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::pointwise_operation({x},{y},{}, "y0 = x0 < 0 ? -x0 : x0;", getExecutionContext(self));

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;

    }

    Tensor abs(const Tensor & self)
    {

        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        Tensor out = new_tensor_as(x.shape(),self);
        dlprim::Tensor y=todp(out);
        dlprim::core::pointwise_operation({x},{y},{},"y0 = x0 < 0 ? -x0 : x0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
