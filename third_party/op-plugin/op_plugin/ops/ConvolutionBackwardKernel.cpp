#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

namespace {
    dlprim::core::Conv2DSettings conv_config(bool transposed,dlprim::Tensor &X,dlprim::Tensor &W,
                    IntArrayRef padding,IntArrayRef stride,IntArrayRef dilation,IntArrayRef output_padding,int groups)
    {
        TORCH_CHECK(stride.size()==2 && padding.size() == 2 && dilation.size() == 2,"Expecting size of parameters=2");
        if(transposed)
            TORCH_CHECK(output_padding.size() == 2,"Expecting transposed size == 2")
        dlprim::Convolution2DConfigBase cfg_base;
        if(!transposed) {
            cfg_base.channels_in = W.shape()[1] * groups;
            cfg_base.channels_out = W.shape()[0];
        }
        else {
            cfg_base.channels_out = W.shape()[1] * groups;
            cfg_base.channels_in = W.shape()[0];
        }
        for(int i=0;i<2;i++) {
            cfg_base.kernel[i] = W.shape()[i+2];
            cfg_base.pad[i] = padding[i];
            cfg_base.stride[i] = stride[i];
            cfg_base.dilate[i] = dilation[i];
            cfg_base.groups = groups;
        }
        if(!transposed) {
            return dlprim::core::Conv2DSettings(cfg_base,X.shape(),X.dtype()); 
        }
        else {
            int op[2] = {int(output_padding[0]),int(output_padding[1])};
            return dlprim::core::Conv2DSettings(cfg_base,dlprim::core::Conv2DBase::get_output_shape_transposed(cfg_base,X.shape(),op),X.dtype());
        }
    }
}


    ::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(
        const at::Tensor & grad_output,
        const at::Tensor & input,
        const at::Tensor & weight,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef dilation,
        bool transposed,
        IntArrayRef output_padding,
        int64_t groups,
        ::std::array<bool,3> output_mask)
    {
        GUARD;
        at::Tensor grad_output_c = grad_output.contiguous();
        at::Tensor input_c = input.contiguous();
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor x  = todp(input_c);
        dlprim::Tensor W  = todp(weight);
        dlprim::core::Conv2DSettings cfg = conv_config(transposed,x,W,padding,stride,dilation,output_padding,groups);
        dlprim::ExecutionContext q = getExecutionContext(input);
        dlprim::Context ctx(q);

        size_t ws_size = 0;
        std::unique_ptr<dlprim::core::Conv2DBackwardData> bwd_data;
        std::unique_ptr<dlprim::core::Conv2DForward> bwd_data_tr;
        std::unique_ptr<dlprim::core::Conv2DBackwardFilter> bwd_filter;
        std::unique_ptr<dlprim::core::BiasBackwardFilter> bwd_bias;

        at::Tensor data_diff,filter_diff,bias_diff;

        if(transposed)
            std::swap(cfg.channels_out,cfg.channels_in);

        if(output_mask[0]) {
            if(!transposed) {
                bwd_data = std::move(dlprim::core::Conv2DBackwardData::create(ctx,cfg)); 
                ws_size = std::max(ws_size,bwd_data->workspace());
            }
            else {
                bwd_data_tr = std::move(dlprim::core::Conv2DForward::create(ctx,cfg,false));
                ws_size = std::max(ws_size,bwd_data_tr->workspace());
            }
        }
        if(output_mask[1]) {
            bwd_filter = std::move(dlprim::core::Conv2DBackwardFilter::create(ctx,cfg)); 
            ws_size = std::max(ws_size,bwd_filter->workspace());
        }
        if(output_mask[2]) {
            bwd_bias = std::move(dlprim::core::BiasBackwardFilter::create(ctx,dy.shape(),dy.dtype()));
            ws_size = std::max(ws_size,bwd_bias->workspace());
        }
        at::DataPtr ws_ptr;
        dlprim::Tensor ws = make_workspace(ws_ptr,ws_size,input.device());

        if(output_mask[0]) {
            data_diff = new_tensor_as(x.shape(),input);
            dlprim::Tensor dx = todp(data_diff);
            if(!transposed)
                bwd_data->enqueue(dx,W,dy,ws,0,q);
            else 
                bwd_data_tr->enqueue(dy,W,nullptr,dx,ws,0,q);
        }

        if(output_mask[1]) {
            filter_diff = new_tensor_as(W.shape(),weight);
            dlprim::Tensor dW = todp(filter_diff);
            if(!transposed)
                bwd_filter->enqueue(x,dW,dy,ws,0,q);
            else
                bwd_filter->enqueue(dy,dW,x,ws,0,q);
        }

        if(output_mask[2]) {
            bias_diff = new_tensor_as(dlprim::Shape(dy.shape()[1]),weight);
            dlprim::Tensor dB = todp(bias_diff);
            bwd_bias->enqueue(dy,dB,ws,0,q);
        }
        
        sync_if_needed(grad_output.device());

        return std::tuple<at::Tensor,at::Tensor,at::Tensor>(data_diff,filter_diff,bias_diff);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */