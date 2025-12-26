#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps)
{
    int N = 1;
    for (auto v : normalized_shape)
    {
        N *= v;
    }

    bool weight_present = weight && weight->numel() > 0;
    bool bias_present = bias && bias->numel() > 0;

    dlprim::ExecutionContext q = getExecutionContext(input);
    dlprim::Context ctx(q);

    at::Tensor input_c = input.contiguous();
    dlprim::Tensor X = todp(input_c);
    TORCH_CHECK(X.shape().total_size() % N == 0, "Invalid input shape");
    int B = X.shape().total_size() / N;
    auto bn_shape = dlprim::Shape(1, B, N);
    auto src_shape = X.shape();
    at::Tensor result = new_tensor_as(X.shape(), input);
    dlprim::Tensor Y = todp(result);
    X.reshape(bn_shape);
    Y.reshape(bn_shape);

    at::Tensor calc_mean_pt, calc_var_pt, calc_rstd_pt;
    dlprim::Tensor calc_mean, calc_var, calc_rstd;

    calc_mean_pt = new_tensor_as(dlprim::Shape(B), input);
    calc_mean = todp(calc_mean_pt);
    calc_var_pt = new_tensor_as(dlprim::Shape(B), input);
    calc_var = todp(calc_var_pt);
    calc_rstd_pt = new_tensor_as(dlprim::Shape(B), input);
    calc_rstd = todp(calc_rstd_pt);

    auto bn = dlprim::core::BatchNormFwdBwd::create(ctx, X.shape(), X.dtype());
    size_t ws_size = bn->workspace();

    DataPtr tmp;
    dlprim::Tensor ws = make_workspace(tmp, ws_size, input.device());

    dlprim::Tensor fwd_mean, fwd_var;

    bn->enqueue_calculate_batch_stats(X, calc_mean, calc_var, ws, q);
    bn->enqueue_forward_get_rstd(X, Y, calc_mean, calc_var, eps, calc_rstd, ws, q);

    Y.reshape(src_shape);
    if (weight_present && bias_present)
    {
        dlprim::Tensor w = todp(*weight);
        dlprim::Tensor b = todp(*bias);
        dlprim::core::pointwise_operation_broadcast({Y, w, b}, {Y}, {}, "y0 = x0 * x1 + x2;", q);
    }
    else if (weight_present)
    {
        dlprim::Tensor w = todp(*weight);
        dlprim::core::pointwise_operation_broadcast({Y, w}, {Y}, {}, "y0 = x0 * x1;", q);
    }
    else if (bias_present)
    {
        dlprim::Tensor b = todp(*bias);
        dlprim::core::pointwise_operation_broadcast({Y, b}, {Y}, {}, "y0 = x0 + x1;", q);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(result, calc_mean_pt, calc_rstd_pt);
}

} /* namespace op_plugin */
} /* namespace at_torch */
