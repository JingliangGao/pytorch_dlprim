#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    // at::Tensor relu(const at::Tensor & self)
    // {
    //     GUARD;
    //     at::Tensor self_c = self.contiguous();
    //     dlprim::Tensor x = todp(self_c);
    //     at::Tensor out = new_tensor_as(x.shape(), self);
    //     dlprim::Tensor y = todp(out);
    //     dlprim::core::activation_forward(x,y,dlprim::StandardActivations::relu, getExecutionContext(self));
    //     sync_if_needed(self.device());
    //     return out;
    // }


    at::Tensor & relu_(at::Tensor & self)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::ExecutionContext q = getExecutionContext(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::relu,q);

        if (!self.is_contiguous())
            self.copy_(self_c);

        sync_if_needed(self.device());
        return self;
    }

    template<dlprim::StandardActivations Act>
    class act_cls : public torch::autograd::Function<act_cls<Act> > {
    public:
        static torch::Tensor std_activation_forward(AutogradContext *ctx, torch::Tensor x)
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;

            Tensor x_c = x.contiguous();
            dlprim::Tensor X = todp(x_c);
            torch::Tensor result = new_tensor_as(X.shape(),x);
            ctx->save_for_backward({result});
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(x);
            dlprim::core::activation_forward(X,Y,Act,q);
            sync_if_needed(x.device());
            return result;
        }
        static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x)
        {
            return std_activation_forward(ctx,x);
        }
        static tensor_list std_activation_backward(AutogradContext *ctx, tensor_list grad_outputs) {
            GUARD;
            auto grad_output = grad_outputs[0].contiguous();
            torch::Tensor result = ctx->get_saved_variables()[0];
            dlprim::Tensor dy=todp(grad_output);
            dlprim::Tensor y=todp(result);
            torch::Tensor grad_input = new_tensor_as(dy.shape(),grad_output);
            dlprim::Tensor dx = todp(grad_input);
            dlprim::core::activation_backward(dx,dy,y,Act,0.0,getExecutionContext(grad_output));
            sync_if_needed(grad_output.device());
            return {grad_input};
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            return std_activation_backward(ctx,grad_outputs);
        }
    };

    template<dlprim::StandardActivations Act>
    torch::Tensor act_autograd(torch::Tensor const &x) {
        GUARD;
        return act_cls<Act>::apply(x);
    }

    at::Tensor relu(const at::Tensor & self) {
         // forward through autograd-enabled wrapper
         return at_torch::op_plugin::act_autograd<dlprim::StandardActivations::relu>(self);
    }



    }  /* namespace op_plugin */
}  /* namespace at_torch */
