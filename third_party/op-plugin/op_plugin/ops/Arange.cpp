#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & arange_out(const Scalar & start, const Scalar & end, const Scalar & step, at::Tensor & out)
    {

        double dstart = start.to<double>();
        double dend   = end.to<double>();
        double dstep = step.to<double>();
        int64_t size = std::ceil((dend - dstart) / dstep);
        int64_t numel = out.numel();

        if(numel  != size) {
            if(numel!=0) {
                 TORCH_WARN("The number of elements in the out tensor of shape ", out.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
            }
            //at::Tensor tmp = new_tensor_as(dlprim::Shape(size),out);
            //out = std::move(tmp);
            out.resize_({size});
        }
        at::Tensor out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        auto q = getExecutionContext(out);
        dlprim::core::pointwise_operation({},{Y},{dstart,dstep},
            "y0 = w0 + index*w1;",q);
        if(!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(out.device());
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
