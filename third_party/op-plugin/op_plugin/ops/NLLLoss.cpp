#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    ::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(
        const at::Tensor & self,
        const at::Tensor & target,
        const c10::optional<at::Tensor> & weight,
        int64_t reduction,
        int64_t ignore_index,
        at::Tensor & output,
        at::Tensor & total_weight)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        at::Tensor target_c = target.contiguous();
        dlprim::Tensor lbl=todp(target_c);
        dlprim::Tensor y=todp(output);
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
            case 0: reduce=false; break; // None
            case 1: reduce=true; scale = 1.0f/x.shape()[0]; break; // Mean
            case 2: reduce=true; break; // sum
        }
        dlprim::core::nll_loss_forward(x,lbl,y,reduce,scale,getExecutionContext(self));
        sync_if_needed(self.device());
        return std::tuple<at::Tensor &,at::Tensor &>(output,total_weight);
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */