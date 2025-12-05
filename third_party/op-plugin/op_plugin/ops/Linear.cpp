#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    class linear_cls : public torch::autograd::Function<linear_cls> {
    public:
        static at::Tensor forward(AutogradContext *ctx,const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias)
        {
            return linear_forward(ctx,input,weight,bias);
        }
        static at::Tensor linear_forward(AutogradContext *ctx,const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias)
        {

            at::AutoDispatchBelowADInplaceOrView g;

            at::Tensor cinput = input.contiguous();
            dlprim::Tensor X = todp(cinput);
            dlprim::Tensor W = todp(weight);
            dlprim::Shape os = X.shape();

            int fi = W.shape()[1];
            int fo = W.shape()[0];
            int batch = X.shape().total_size()/fi;

            os[os.size()-1] = fo;

            at::Tensor result = new_tensor_as(os,input);
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(input);
            dlprim::Context dlprim_ctx(q);
            dlprim::core::IPSettings cfg;
            cfg.inputs = fi;
            cfg.outputs = fo;
            cfg.optimal_batch_size = batch;
            cfg.dtype = todp(input.dtype());
            bool has_bias = bias && bias->numel() > 0;
            auto ip = dlprim::core::IPForward::create(dlprim_ctx,cfg,has_bias);
            dlprim::Tensor B;
            if(has_bias)
                B=todp(*bias);
            X.reshape(dlprim::Shape(batch,fi));
            Y.reshape(dlprim::Shape(batch,fo));
            ip->enqueue(X,W,(has_bias ? &B : nullptr),Y,q);
            ctx->save_for_backward({cinput,weight});
            ctx->saved_data["has_bias"]=has_bias;

            sync_if_needed(input.device());
            return result;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            return linear_backward(ctx,grad_outputs);
        }
        static tensor_list linear_backward(AutogradContext *ctx, tensor_list grad_outputs) {

            dlprim::Tensor X = todp(ctx->get_saved_variables()[0]);
            dlprim::Tensor W = todp(ctx->get_saved_variables()[1]);

            int fi = W.shape()[1];
            int fo = W.shape()[0];
            int batch = X.shape().total_size()/fi;

            at::Tensor dy_tensor = grad_outputs[0].contiguous();
            dlprim::Tensor dY = todp(dy_tensor);
            auto grad_output = grad_outputs[0];

            at::Tensor dx_tensor = new_tensor_as(X.shape(),dy_tensor);
            dlprim::Tensor dX = todp(dx_tensor);

            at::Tensor dW_tensor = new_tensor_as(W.shape(),dy_tensor);
            dlprim::Tensor dW = todp(dW_tensor);

            dlprim::core::IPSettings cfg;
            cfg.inputs = fi;
            cfg.outputs = fo;
            cfg.optimal_batch_size = batch;
            cfg.dtype = todp(dx_tensor.dtype());

            auto q = getExecutionContext(dy_tensor);
            dlprim::Context dlprim_ctx(q);

            dlprim::Shape X_shape(batch,fi);
            dlprim::Shape Y_shape(batch,fo);

            X.reshape(X_shape);
            dX.reshape(X_shape);
            dY.reshape(Y_shape);

            auto bwd_data = dlprim::core::IPBackwardData::create(dlprim_ctx,cfg);
            bwd_data->enqueue(dX,W,dY,0,q);

            auto bwd_filter = dlprim::core::IPBackwardFilter::create(dlprim_ctx,cfg);
            bwd_filter->enqueue(X,dW,dY,0,q);

            bool has_bias = ctx->saved_data["has_bias"].toBool();
            at::Tensor dB_tensor;
            if(has_bias) {
                dB_tensor = new_tensor_as(dlprim::Shape(W.shape()[0]),dy_tensor);
                dlprim::Tensor dB=todp(dB_tensor);
                auto bwd_bias = dlprim::core::BiasBackwardFilter::create(dlprim_ctx,dY.shape(),cfg.dtype);
                at::DataPtr ptr;
                dlprim::Tensor ws = make_workspace(ptr,bwd_bias->workspace(),dy_tensor.device());
                bwd_bias->enqueue(dY,dB,ws,0,q);
            }

            sync_if_needed(grad_output.device());
            return {dx_tensor,dW_tensor,dB_tensor};
        }
    };
    at::Tensor linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias)
    {

        return linear_cls::apply(input,weight,bias);
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
