#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor mse_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        at::Tensor target_c = target.contiguous();
        dlprim::Tensor lbl=todp(target_c);
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
            case 0: reduce=false; break; // None
            case 1: reduce=true; scale = 1.0f/x.shape().total_size(); break; // Mean
            case 2: reduce=true; break; // sum
        }
        at::Tensor output = new_tensor_as(reduce ? dlprim::Shape() : x.shape(),self_c);
        dlprim::Tensor y=todp(output);
        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(ctx,
                    {x.specs(),lbl.specs()},{y.specs()},0,x.dtype(),
                    "y0 = (x0-x1)*(x0-x1);",
                    "reduce_y0 = 0;",
                    "reduce_y0 += y0;");
        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({x,lbl},{y},wsg.ws,{},{scale},{0},q);
        sync_if_needed(self.device());

        return output;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
