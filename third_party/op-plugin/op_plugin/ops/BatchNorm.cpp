#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    ::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps)
    {
        GUARD;
        bool weight_present = weight && weight->numel()>0;
        bool bias_present = bias && bias->numel()>0;
        bool mean_present = running_mean && running_mean->numel() > 0;
        bool var_present = running_var && running_var->numel() > 0;
        TORCH_CHECK(weight_present == bias_present,"Can have affince or not affine but not partial")
        bool affine = weight_present && bias_present;
        TORCH_CHECK(mean_present && var_present,"Running sums are expected to be present")
        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);

        at::Tensor input_c = input.contiguous();
        dlprim::Tensor X = todp(input_c);
        at::Tensor result = new_tensor_as(X.shape(),input);
        dlprim::Tensor Y = todp(result);
        dlprim::Tensor gamma,beta;
        if(affine) {
            gamma = todp(*weight);
            beta  = todp(*bias);
        }
        dlprim::Tensor mean = todp(*running_mean);
        dlprim::Tensor var  = todp(*running_var);
        at::Tensor calc_mean_pt,calc_var_pt;
        dlprim::Tensor calc_mean,calc_var;

        if(training) {
            calc_mean_pt = new_tensor_as(mean.shape(),*running_mean);
            calc_mean = todp(calc_mean_pt);
            calc_var_pt  = new_tensor_as(var.shape(),*running_var);
            calc_var = todp(calc_var_pt);
        }

        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,X.shape(),X.dtype());
        size_t ws_size = bn->workspace();

        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

        dlprim::Tensor fwd_mean,fwd_var;

        if(training) {
            size_t M = X.shape().total_size() / X.shape()[1];
            bn->enqueue_calculate_batch_stats(X,calc_mean,calc_var,ws,q);
            bn->enqueue_update_running_stats(
                            momentum,(1.0f-momentum),
                            calc_mean,mean,
                            (momentum * M) / (M-1),(1.0f-momentum),
                            calc_var,var,
                            ws,q);
            fwd_mean = calc_mean;
            fwd_var  = calc_var;
        }
        else {
            fwd_mean = mean;
            fwd_var  = var;
        }
        if(affine) {
            bn->enqueue_forward_affine(X,Y,gamma,beta,fwd_mean,fwd_var,eps,ws,q);
        }
        else {
            bn->enqueue_forward_direct(X,Y,fwd_mean,fwd_var,eps,ws,q);
        }
        return std::tuple<at::Tensor,at::Tensor,at::Tensor>(result,calc_mean_pt,calc_var_pt);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
