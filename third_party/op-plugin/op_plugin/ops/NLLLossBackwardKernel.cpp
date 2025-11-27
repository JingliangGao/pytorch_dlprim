#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & nll_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & /*total_weight*/, at::Tensor & grad_input)
    {
        GUARD;

        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");

        dlprim::Tensor dx=todp(grad_input);
        at::Tensor target_c = target.contiguous();
        at::Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor lbl=todp(target_c);
        dlprim::Tensor dy=todp(grad_output_c);
        
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
            case 0: reduce=false; break; // None
            case 1: reduce=true; scale = 1.0f/dx.shape()[0]; break; // Mean
            case 2: reduce=true; break; // sum
        }
        dlprim::core::nll_loss_backward(dx,lbl,dy,reduce,scale,0.0f,getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */