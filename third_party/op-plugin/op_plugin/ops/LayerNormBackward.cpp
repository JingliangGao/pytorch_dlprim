#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& save_mean,
    const at::Tensor& save_rstd,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    ::std::array<bool, 3> output_mask)
{

    int N = 1;
    std::vector<int> ns;
    for (auto v : normalized_shape)
    {
        ns.push_back(v);
        N *= v;
    }
    dlprim::Shape norm_shape = dlprim::Shape::from_range(ns.begin(), ns.end());

    bool weight_present = weight && weight->numel() > 0;
    bool bias_present = bias && bias->numel() > 0;

    dlprim::ExecutionContext q = getExecutionContext(input);
    dlprim::Context ctx(q);
    at::Tensor grad_out_c = grad_out.contiguous();
    at::Tensor input_c = input.contiguous();
    dlprim::Tensor dY = todp(grad_out_c);
    dlprim::Tensor X = todp(input_c);
    auto src_shape = X.shape();

    int B = X.shape().total_size() / N;
    auto bn_shape = dlprim::Shape(1, B, N);
    X.reshape(bn_shape);
    dY.reshape(bn_shape);

    dlprim::Tensor W;

    if (weight_present)
    {
        W = todp(*weight);
        W.reshape(dlprim::Shape(N));
    }

    at::Tensor x_diff, gamma_diff, beta_diff;

    bool bwd_data = output_mask[0];
    bool bwd_gamma = output_mask[1] && weight_present;
    bool bwd_beta = output_mask[2] && bias_present;

    dlprim::Tensor dX, dG, dB;
    if (bwd_gamma)
    {
        gamma_diff = new_tensor_as(norm_shape, input);
        dG = todp(gamma_diff);
        dG.reshape(dlprim::Shape(N));
        auto mean = todp(save_mean);
        auto rstd = todp(save_rstd);
        mean.reshape(dlprim::Shape(1, B, 1));
        rstd.reshape(dlprim::Shape(1, B, 1));
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
            ctx,
            {X.specs(), mean.specs(), rstd.specs(), dY.specs()},
            {dG.specs()},
            0,
            dlprim::float_data,
            "y0=(x0 - x1)*x2*x3;",
            "reduce_y0 = 0;",
            "reduce_y0 += y0;");
        WSGuard wsg(op->workspace(), input.device());
        op->enqueue({X, mean, rstd, dY}, {dG}, wsg.ws, {}, {1}, {0}, q);
    }
    if (bwd_beta)
    {
        beta_diff = new_tensor_as(norm_shape, input);
        dB = todp(beta_diff);
        dB.reshape(dlprim::Shape(N));
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
            ctx,
            {dY.specs()},
            {dB.specs()},
            0,
            dlprim::float_data,
            "y0=x0;",
            "reduce_y0 = 0;",
            "reduce_y0 += y0;");
        WSGuard wsg(op->workspace(), input.device());
        op->enqueue({dY}, {dB}, wsg.ws, {}, {1}, {0}, q);
    }
    if (bwd_data)
    {
        x_diff = new_tensor_as(src_shape, input);
        dX = todp(x_diff);
        dX.reshape(bn_shape);
        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx, bn_shape, X.dtype());
        size_t ws_size = bn->workspace();

        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp, ws_size, input.device());

        dlprim::Tensor mean = todp(save_mean);
        dlprim::Tensor rstd = todp(save_rstd);
        dlprim::Tensor dYW_diff = dY;

        if (weight_present)
        {
            auto pt_dYW_diff = new_tensor_as(dY.shape(), input);
            dYW_diff = todp(pt_dYW_diff);
            dlprim::core::pointwise_operation_broadcast(
                {dY, W}, {dYW_diff}, {}, {}, "y0 = x0 * x1;", q);
        }

        bn->enqueue_backward_rstd(X, dYW_diff, mean, rstd, dX, 0.0, ws, q);
    }

    sync_if_needed(input.device());
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(x_diff, gamma_diff, beta_diff);
}

} /* namespace op_plugin */
} /* namespace at_torch */
