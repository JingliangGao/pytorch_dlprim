#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & pow_out(const at::Tensor & self, const Scalar & exponent, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();
        double val = exponent.toDouble();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(out_c);

        dlprim::core::pointwise_operation({x},{y},{val}, "y0=pow(x0,w0);", getExecutionContext(self));

        if(!out.is_contiguous())
            out.copy_(out_c);
        sync_if_needed(self.device());
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
