#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor dot(const at::Tensor & self, const at::Tensor & tensor)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor tensor_c = tensor.contiguous();

        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor x1=todp(tensor_c);
        at::Tensor result = new_tensor_as(dlprim::Shape(),self_c);
        dlprim::Tensor y=todp(result);
        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                ctx,
                {x0.specs(),x1.specs()},{y.specs()},
                0,dlprim::float_data,
                "y0=x0*x1;",
                "reduce_y0 = 0;",
                "reduce_y0 += y0;");

        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({x0,x1},{y},wsg.ws,{},{1},{0},q);
        sync_if_needed(self.device());
        return result;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
