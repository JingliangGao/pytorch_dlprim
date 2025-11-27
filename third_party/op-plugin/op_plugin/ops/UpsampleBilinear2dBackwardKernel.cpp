#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

namespace {

    at::Tensor & interpolate_2d_backward_out_internal(const at::Tensor & grad_output, at::IntArrayRef /*output_size*/, at::IntArrayRef /*input_size*/, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input,dlprim::InterpolateType method,bool align_c=false)
    {
        GUARD;
        at::Tensor grad_output_c = grad_output.contiguous();
        at::Tensor grad_input_c  = grad_input.contiguous();
        double scale_h = scales_h ? *scales_h : -1;
        double scale_w = scales_w ? *scales_w : -1;
        dlprim::Tensor dx = todp(grad_input_c);
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::core::interpolate2d_backward(dx,dy,scale_h,scale_w,method,align_c,0.0,getExecutionContext(grad_output));
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        sync_if_needed(grad_output.device());
        return grad_input;
    }
}

    at::Tensor & upsample_bilinear2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input)
    {
        GUARD;
        return interpolate_2d_backward_out_internal(grad_output,output_size,input_size,scales_h,scales_w,grad_input,dlprim::InterpolateType::bilinear,align_corners);
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */