#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    ::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(const at::Tensor & grad_out,
                                                                  const at::Tensor & input,
                                                                  const c10::optional<at::Tensor> & weight,
                                                                  const c10::optional<at::Tensor> & running_mean,
                                                                  const c10::optional<at::Tensor> & running_var,
                                                                  const c10::optional<at::Tensor> & save_mean,
                                                                  const c10::optional<at::Tensor> & save_var,
                                                                  bool train,
                                                                  double eps,
                                                                  ::std::array<bool,3> output_mask)
    {
        GUARD;
        bool weight_present = weight && weight->numel()>0; 
        bool affine = weight_present;
        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);

        dlprim::Tensor dY = todp(grad_out);
        dlprim::Tensor X = todp(input);
        dlprim::Tensor W;
        if(weight_present)
            W = todp(*weight);
        at::Tensor x_diff,gamma_diff,beta_diff;

        bool bwd_data=output_mask[0];
        bool bwd_gamma=output_mask[1] && affine;
        bool bwd_beta=output_mask[2] && affine;
        dlprim::Tensor dX,dG,dB;
        if(bwd_data) {
            x_diff = new_tensor_as(X.shape(),input);
            dX = todp(x_diff);
        }
        if(bwd_gamma)  {
            gamma_diff = new_tensor_as(dlprim::Shape(X.shape()[1]),input);
            dG = todp(gamma_diff);
        }
        if(bwd_beta) {
            beta_diff = new_tensor_as(dlprim::Shape(X.shape()[1]),input);
            dB = todp(beta_diff);
        }

        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,X.shape(),X.dtype());
        size_t ws_size = bn->workspace();
        
        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

        dlprim::Tensor mean = train ? todp(*save_mean) : todp(*running_mean);
        dlprim::Tensor var  = train ? todp(*save_var)  : todp(*running_var);

        if(affine) {
            bn->enqueue_backward_affine(
                    train,
                    X,dY,
                    mean,var,
                    W, 
                    (bwd_data  ? &dX : nullptr),
                    0.0,
                    (bwd_gamma ? &dG : nullptr),
                    0.0, 
                    (bwd_beta  ? &dB : nullptr),
                    0.0,
                    eps,
                    ws,q);
        }
        else {
            bn->enqueue_backward_direct(
                    train,
                    X,dY,
                    mean,var,
                    dX,0.0,
                    eps,
                    ws,q);

        }
        sync_if_needed(input.device());
        return std::tuple<at::Tensor,at::Tensor,at::Tensor>(x_diff,gamma_diff,beta_diff);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */