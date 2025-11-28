#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & add_out(const at::Tensor & self, const at::Tensor & other, const Scalar & alpha, at::Tensor & out)
    {
        GUARD;
        at::Tensor out_c = out.contiguous();
        dlprim::Tensor y0=todp(out_c);
        double value=0;
        auto dev_to_sync = self.device();
        if(is_cpu_scalar(other,value)) {
            at::Tensor self_c = self.contiguous();
            dlprim::Tensor x0=todp(self_c);
            float w0 = alpha.toDouble() * value;
            dlprim::core::pointwise_operation({x0},{y0},{w0},
                                      "y0 = x0 + w0;",
                                      getExecutionContext(self));
        }
        else if(is_cpu_scalar(self,value)) {
            dev_to_sync = other.device();
            at::Tensor other_c = other.contiguous();
            dlprim::Tensor x0=todp(other_c);
            float w0 = value;
            float w1 = alpha.toDouble();
            dlprim::core::pointwise_operation({x0},{y0},{w0,w1},
                                      "y0 = w0 + x0 * w1;",
                                      getExecutionContext(other));
        }
        else {
            at::Tensor self_c = self.contiguous();
            dlprim::Tensor x0=todp(self_c);
            at::Tensor other_c = other.contiguous();
            dlprim::Tensor x1=todp(other_c);
            float w0 = alpha.toDouble();
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{w0},
                                      "y0 = x0 + x1 * w0;",
                                      getExecutionContext(self));
        }
        if (!out.is_contiguous())
            out.copy_(out_c);
        sync_if_needed(dev_to_sync);
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
