#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

namespace {

    at::Tensor & interpolate_2d_out_internal(const at::Tensor & self, at::IntArrayRef /*output_size*/, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out,dlprim::InterpolateType method,bool align_c=false)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();
        double scale_h = scales_h ? *scales_h : -1;
        double scale_w = scales_w ? *scales_w : -1;
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(out_c);
        dlprim::core::interpolate2d(x,y,scale_h,scale_w,method,align_c,getExecutionContext(self));
        if(!out.is_contiguous())
            out.copy_(out_c);
        sync_if_needed(self.device());
        return out;
    }
}

   
    at::Tensor & _upsample_nearest_exact2d_out(const at::Tensor & self, at::IntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out)
    {
        GUARD;
        return interpolate_2d_out_internal(self,output_size,scales_h,scales_w,out,dlprim::InterpolateType::nearest_exact);
    }
    

    }  /* namespace op_plugin */
}  /* namespace at_torch */