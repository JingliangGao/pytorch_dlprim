#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const Scalar &sbeta, const Scalar & salpha, at::Tensor & out)
    {
        GUARD;
        double alpha = salpha.toDouble();
        double beta = sbeta.toDouble();
        if(alpha == 1 && beta == 0) {
            return mm_out(mat1,mat2,out);
        }
        at::Tensor p = torch::mm(mat1,mat2);
        if(beta == 0) {
            auto x=todp(p);
            if(out.is_contiguous()) {
                auto y=todp(out);
                dlprim::core::pointwise_operation({x},{y},{alpha},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
            }
            else {
                dlprim::core::pointwise_operation({x},{x},{alpha},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
                out.copy_(p);
            }
        }
        else {
            dlprim::Tensor x = todp(p);
            at::Tensor self_c = self.contiguous();
            dlprim::Tensor off = todp(self_c);
            dlprim::Tensor tgt;
            if(out.is_contiguous()) {
                tgt = todp(out);
            }
            else {
                tgt = x;
            }
            dlprim::core::pointwise_operation_broadcast({x,off},{tgt},{alpha,beta},
                    "y0 = x0 * w0 + x1 * w1;",
                    getExecutionContext(self));
            if(!out.is_contiguous()) {
                out.copy_(p);
            }
        }
        sync_if_needed(self.device());
        return out;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
