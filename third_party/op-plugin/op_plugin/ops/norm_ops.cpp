#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    // {"schema": "aten::native_batch_norm(at::Tensor input, at::Tensor? weight, at::Tensor? bias, at::Tensor? running_mean, at::Tensor? running_var, bool training, float momentum, float eps) -> (at::Tensor, at::Tensor, at::Tensor)", "dispatch": "True", "default": "False"}
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

    // {"schema": "aten::native_batch_norm_backward(at::Tensor grad_out, at::Tensor input, at::Tensor? weight, at::Tensor? running_mean, at::Tensor? running_var, at::Tensor? save_mean, at::Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (at::Tensor, at::Tensor, at::Tensor)", "dispatch": "True", "default": "False"} 
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

    // {"schema": "aten::native_layer_norm(at::Tensor input, SymInt[] normalized_shape, at::Tensor? weight, at::Tensor? bias, float eps) -> (at::Tensor, at::Tensor, at::Tensor)", "dispatch": "True", "default": "True"}
    std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps)
    {
        int N = 1;
        for(auto v:normalized_shape) {
            N *= v;
        }

        bool weight_present = weight && weight->numel()>0; 
        bool bias_present = bias && bias->numel()>0; 

        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);

        at::Tensor input_c = input.contiguous();
        dlprim::Tensor X = todp(input_c);
        TORCH_CHECK(X.shape().total_size() % N == 0,"Invalid input shape");
        int B = X.shape().total_size() / N;
        auto bn_shape = dlprim::Shape(1,B,N);
        auto src_shape = X.shape();
        at::Tensor result = new_tensor_as(X.shape(),input);
        dlprim::Tensor Y = todp(result);
        X.reshape(bn_shape);
        Y.reshape(bn_shape);


        at::Tensor calc_mean_pt,calc_var_pt,calc_rstd_pt;
        dlprim::Tensor calc_mean,calc_var,calc_rstd;

        calc_mean_pt = new_tensor_as(dlprim::Shape(B),input);
        calc_mean = todp(calc_mean_pt);
        calc_var_pt  = new_tensor_as(dlprim::Shape(B),input);
        calc_var = todp(calc_var_pt);
        calc_rstd_pt  = new_tensor_as(dlprim::Shape(B),input);
        calc_rstd = todp(calc_rstd_pt);

        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,X.shape(),X.dtype());
        size_t ws_size = bn->workspace();
        
        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

        dlprim::Tensor fwd_mean,fwd_var;

        bn->enqueue_calculate_batch_stats(X,calc_mean,calc_var,ws,q);
        bn->enqueue_forward_get_rstd(X,Y,calc_mean,calc_var,eps,calc_rstd,ws,q);

        Y.reshape(src_shape);
        if(weight_present && bias_present) {
            dlprim::Tensor w = todp(*weight);
            dlprim::Tensor b = todp(*bias);
            dlprim::core::pointwise_operation_broadcast({Y,w,b},{Y},{},
                                      "y0 = x0 * x1 + x2;",
                                      q);

        }
        else if(weight_present) {
            dlprim::Tensor w = todp(*weight);
            dlprim::core::pointwise_operation_broadcast({Y,w},{Y},{},
                                      "y0 = x0 * x1;",
                                      q);
        }
        else if(bias_present) {
            dlprim::Tensor b = todp(*bias);
            dlprim::core::pointwise_operation_broadcast({Y,b},{Y},{},
                                      "y0 = x0 + x1;",
                                      q);
        }
        return std::tuple<at::Tensor,at::Tensor,at::Tensor>(result,calc_mean_pt,calc_rstd_pt);
    }
    // {"schema": "aten::native_layer_norm_backward(at::Tensor grad_out, at::Tensor input, SymInt[] normalized_shape, at::Tensor mean, at::Tensor rstd, at::Tensor? weight, at::Tensor? bias, bool[3] output_mask) -> (at::Tensor, at::Tensor, at::Tensor)", "dispatch": "True", "default": "False"}
    std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(
            const at::Tensor & grad_out,
            const at::Tensor & input,
            at::IntArrayRef normalized_shape,
            const at::Tensor & save_mean,
            const at::Tensor & save_rstd,
            const c10::optional<at::Tensor> & weight,
            const c10::optional<at::Tensor> & bias,
            ::std::array<bool,3> output_mask)
    {
        GUARD;
        int N = 1;
        std::vector<int> ns;
        for(auto v:normalized_shape) {
            ns.push_back(v);
            N *= v;
        }
        dlprim::Shape norm_shape = dlprim::Shape::from_range(ns.begin(),ns.end());


        bool weight_present = weight && weight->numel()>0; 
        bool bias_present = bias && bias->numel() > 0;

        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);
        at::Tensor grad_out_c = grad_out.contiguous();
        at::Tensor input_c = input.contiguous();
        dlprim::Tensor dY = todp(grad_out_c);
        dlprim::Tensor X = todp(input_c);
        auto src_shape = X.shape();

        int B = X.shape().total_size() / N;
        auto bn_shape = dlprim::Shape(1,B,N);
        X.reshape(bn_shape);
        dY.reshape(bn_shape);
        
        dlprim::Tensor W;
        
        if(weight_present) {
            W = todp(*weight);
            W.reshape(dlprim::Shape(N));
        }

        at::Tensor x_diff,gamma_diff,beta_diff;

        bool bwd_data=output_mask[0];
        bool bwd_gamma=output_mask[1] && weight_present;
        bool bwd_beta=output_mask[2] && bias_present;

        
        dlprim::Tensor dX,dG,dB;
        if(bwd_gamma)  {
            gamma_diff = new_tensor_as(norm_shape,input);
            dG = todp(gamma_diff);
            dG.reshape(dlprim::Shape(N));
            auto mean = todp(save_mean);
            auto rstd = todp(save_rstd);
            mean.reshape(dlprim::Shape(1,B,1));
            rstd.reshape(dlprim::Shape(1,B,1));
            auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                        ctx,
                        {X.specs(),mean.specs(),rstd.specs(),dY.specs()},{dG.specs()},
                        0,dlprim::float_data,
                        "y0=(x0 - x1)*x2*x3;",
                        "reduce_y0 = 0;",
                        "reduce_y0 += y0;");
            WSGuard wsg(op->workspace(),input.device());
            op->enqueue({X,mean,rstd,dY},{dG},wsg.ws,{},{1},{0},q);
        }
        if(bwd_beta) {
            beta_diff = new_tensor_as(norm_shape,input);
            dB = todp(beta_diff);
            dB.reshape(dlprim::Shape(N));
            auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                        ctx,
                        {dY.specs()},{dB.specs()},
                        0,dlprim::float_data,
                        "y0=x0;",
                        "reduce_y0 = 0;",
                        "reduce_y0 += y0;");
            WSGuard wsg(op->workspace(),input.device());
            op->enqueue({dY},{dB},wsg.ws,{},{1},{0},q);
        }
        if(bwd_data) {
            x_diff = new_tensor_as(src_shape,input);
            dX = todp(x_diff);
            dX.reshape(bn_shape);
            auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,bn_shape,X.dtype());
            size_t ws_size = bn->workspace();
            
            DataPtr tmp;
            dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

            dlprim::Tensor mean = todp(save_mean);
            dlprim::Tensor rstd  = todp(save_rstd);
            dlprim::Tensor dYW_diff = dY;

            if(weight_present) {
                auto pt_dYW_diff = new_tensor_as(dY.shape(),input);
                dYW_diff = todp(pt_dYW_diff);
                dlprim::core::pointwise_operation_broadcast({dY,W},{dYW_diff},{},{},
                        "y0 = x0 * x1;", 
                        q);
            }

            bn->enqueue_backward_rstd(
                    X,dYW_diff,
                    mean,rstd,
                    dX,0.0,
                    ws,q);
        }

        sync_if_needed(input.device());
        return std::tuple<at::Tensor,at::Tensor,at::Tensor>(x_diff,gamma_diff,beta_diff);
    }

}  /* namespace op_plugin */
}  /* namespace at_torch */



