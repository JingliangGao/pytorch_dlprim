#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max, at::Tensor & out)
    {

        at::Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        dlprim::Tensor X = todp(self_c);
        auto q = getExecutionContext(self);
        if(min && max)
            dlprim::core::pointwise_operation({X},{Y},{min->to<double>(),max->to<double>()},"y0 = max(w0,min(w1,x0));",q);
        else if(min)
            dlprim::core::pointwise_operation({X},{Y},{min->to<double>()},"y0 = max(w0,x0);",q);
        else if(max)
            dlprim::core::pointwise_operation({X},{Y},{max->to<double>()},"y0 = min(w0,x0);",q);
        else
            dlprim::core::pointwise_operation({X},{Y},{},"y0 = x0;",q);

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }


    at::Tensor & clamp_min_out(const at::Tensor & self, const Scalar & min, at::Tensor & out)
    {

        at::Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        dlprim::Tensor X = todp(self_c);
        auto q = getExecutionContext(self);

        dlprim::core::pointwise_operation({X},{Y},{min.to<double>()},"y0 = max(w0,x0);",q);

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
