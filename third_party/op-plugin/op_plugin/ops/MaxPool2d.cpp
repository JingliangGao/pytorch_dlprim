#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    class max_pool2d_cls : public torch::autograd::Function<max_pool2d_cls> {
    public:
        static at::Tensor forward(AutogradContext *ctx,at::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
        {
            return max_pool2d_forward(ctx,self,kernel_size,stride,padding,dilation,ceil_mode);
        }

        static at::Tensor max_pool2d_forward(AutogradContext *ctx,at::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;

            TORCH_CHECK(kernel_size.size()==2,"Invalid sizes");
            TORCH_CHECK(dilation[0]==1 && dilation[1]==1,"Dilation != 1 is not implemented yet");
            int kernel[2]={int(kernel_size[0]),int(kernel_size[1])};
            int pad[2]={int(padding[0]),int(padding[1])};
            int strd[2]={1,1};
            if(stride.size()!=0) {
                TORCH_CHECK(stride.size()==2);
                strd[0] = stride[0];
                strd[1] = stride[1];
            }
            else {
                strd[0] = kernel[0];
                strd[1] = kernel[1];
            }

            at::Tensor self_cont = self.contiguous();
            dlprim::Tensor X = todp(self_cont);
            dlprim::Shape x_shape = X.shape();
            dlprim::Shape y_shape = dlprim::Shape(
                    x_shape[0],
                    x_shape[1],
                    dlprim::core::calc_pooling_output_size(x_shape[2],kernel[0],pad[0],strd[0],ceil_mode),
                    dlprim::core::calc_pooling_output_size(x_shape[3],kernel[1],pad[1],strd[1],ceil_mode));

            at::Tensor out = new_tensor_as(y_shape,self);

            dlprim::Tensor Y = todp(out);
            dlprim::ExecutionContext q = getExecutionContext(self);
            dlprim::Context dlprim_ctx(q);
            auto pool = dlprim::core::Pooling2DForward::create_max_pooling(dlprim_ctx,kernel,pad,strd,todp(self.dtype()));
            pool->enqueue(X,Y,q);
            sync_if_needed(self.device());

            ctx->save_for_backward({self_cont});
            ctx->saved_data["kernel_0"]=kernel[0];
            ctx->saved_data["kernel_1"]=kernel[1];
            ctx->saved_data["pad_0"]=pad[0];
            ctx->saved_data["pad_1"]=pad[1];
            ctx->saved_data["strd_0"]=strd[0];
            ctx->saved_data["strd_1"]=strd[1];

            return out;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            return max_pool_2d_backward(ctx,grad_outputs);
        }
        static tensor_list max_pool_2d_backward(AutogradContext *ctx, tensor_list grad_outputs) {
            GUARD;
            at::Tensor grad_output = grad_outputs[0];
            at::Tensor input = ctx->get_saved_variables()[0];
            int kernel[2],pad[2],strd[2];
            kernel[0] = ctx->saved_data["kernel_0"].toInt();
            kernel[1] = ctx->saved_data["kernel_1"].toInt();

            pad[0] = ctx->saved_data["pad_0"].toInt();
            pad[1] = ctx->saved_data["pad_1"].toInt();

            strd[0] = ctx->saved_data["strd_0"].toInt();
            strd[1] = ctx->saved_data["strd_1"].toInt();

            at::Tensor grad_output_c = grad_output.contiguous(),input_c = input.contiguous();
            dlprim::Tensor dy=todp(grad_output_c);
            dlprim::Tensor x=todp(input_c);
            at::Tensor grad_input = new_tensor_as(x.shape(),grad_output);
            dlprim::Tensor dx = todp(grad_input);

            dlprim::ExecutionContext q = getExecutionContext(grad_output);
            dlprim::Context dlprim_ctx(q);

            auto pool=dlprim::core::MaxPooling2DBackward::create(dlprim_ctx,kernel,pad,strd,todp(input.dtype()));
            pool->enqueue(x,dx,dy,0,q);
            sync_if_needed(grad_output.device());
            return {grad_input,at::Tensor(),at::Tensor(),at::Tensor(),at::Tensor(),at::Tensor()};
        }
    };

    at::Tensor max_pool2d(at::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
        GUARD;
        return max_pool2d_cls::apply(self,kernel_size,stride,padding,dilation,ceil_mode);
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
