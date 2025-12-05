#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & logit_out(const at::Tensor & self, ::std::optional<double> eps, at::Tensor & out)
    {

        at::Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor Y = todp(out_c);
        auto q = getExecutionContext(self);
        if(eps) {
            double e = *eps;
            dlprim::core::pointwise_operation({X},{Y},{e},
                "dtype z = min(1.0f-w0,max(w0,x0)); "
                "y0 = log(z / (z-1.0f)); ",q);
        }
        else {
            dlprim::core::pointwise_operation({X},{Y},{},
                "y0 = log(x0 / (x0-1.0f));",q);
        }
        if(!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }


    at::Tensor logit(const at::Tensor & self, ::std::optional<double> eps)
    {
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);

        at::Tensor result = new_tensor_as(X.shape(),self);
        logit_out(self_c,eps,result);
        return result;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
