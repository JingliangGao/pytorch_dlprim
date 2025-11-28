#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & avg_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input)
    {
        GUARD;
        TORCH_CHECK(ceil_mode==false,"Ceil mode=true not implemented");
        TORCH_CHECK(!divisor_override,"Divisor override is not implemented");
        int ker[2] = {int(kernel_size[0]),int(kernel_size[1])};
        int pad[2] = {int(padding[0]),    int(padding[1])};
        int strd[2];
        if(stride.empty()) {
            strd[0]=ker[0];
            strd[1]=ker[1];
        }
        else {
            strd[0]=stride[0];
            strd[1]=stride[1];
        };
        at::Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor dY=todp(grad_output_c);
        dlprim::Tensor dX=todp(grad_input);
        dlprim::ExecutionContext q(getExecutionContext(self));
        dlprim::Context ctx(q);

        auto pool = dlprim::core::AvgPooling2DBackward::create(
                        ctx,
                        ker,pad,strd,
                        count_include_pad,todp(grad_input.dtype())
                    );
        pool->enqueue(dX,dY,0,q);
        sync_if_needed(self.device());
        return grad_input;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
