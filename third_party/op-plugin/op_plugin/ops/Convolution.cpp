#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

namespace
{
dlprim::core::Conv2DSettings conv_config(
    bool transposed,
    dlprim::Tensor& X,
    dlprim::Tensor& W,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef output_padding,
    int groups)
{
    TORCH_CHECK(
        stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
        "Expecting size of parameters=2");
    if (transposed)
        TORCH_CHECK(output_padding.size() == 2, "Expecting transposed size == 2")
    dlprim::Convolution2DConfigBase cfg_base;
    if (!transposed)
    {
        cfg_base.channels_in = W.shape()[1] * groups;
        cfg_base.channels_out = W.shape()[0];
    }
    else
    {
        cfg_base.channels_out = W.shape()[1] * groups;
        cfg_base.channels_in = W.shape()[0];
    }
    for (int i = 0; i < 2; i++)
    {
        cfg_base.kernel[i] = W.shape()[i + 2];
        cfg_base.pad[i] = padding[i];
        cfg_base.stride[i] = stride[i];
        cfg_base.dilate[i] = dilation[i];
        cfg_base.groups = groups;
    }
    if (!transposed)
    {
        return dlprim::core::Conv2DSettings(cfg_base, X.shape(), X.dtype());
    }
    else
    {
        int op[2] = {int(output_padding[0]), int(output_padding[1])};
        return dlprim::core::Conv2DSettings(
            cfg_base,
            dlprim::core::Conv2DBase::get_output_shape_transposed(cfg_base, X.shape(), op),
            X.dtype());
    }
}
} // namespace

at::Tensor convolution_overrideable(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups)
{

    at::Tensor X_tmp = input.contiguous();
    dlprim::Tensor X = todp(X_tmp);
    dlprim::Tensor W = todp(weight);
    dlprim::Tensor B;
    TORCH_CHECK(X.shape().size() == 4, "Invalid input shape");
    TORCH_CHECK(W.shape().size() == 4, "Invalid input shape");
    bool with_bias = bias && bias->numel() != 0;
    if (with_bias)
    {
        B = todp(*bias);
    }

    dlprim::core::Conv2DSettings cfg =
        conv_config(transposed, X, W, padding, stride, dilation, output_padding, groups);

    dlprim::ExecutionContext q = getExecutionContext(input);
    dlprim::Context ctx(q);
    at::Tensor result;
    if (!transposed)
    {
        auto conv = dlprim::core::Conv2DForward::create(ctx, cfg, with_bias);
        WSGuard wsg(conv->workspace(), input.device());

        dlprim::Shape rs = dlprim::core::Conv2DForward::get_output_shape(cfg, X.shape());
        result = new_tensor_as(rs, input);
        dlprim::Tensor Y = todp(result);
        conv->enqueue(X, W, (with_bias ? &B : nullptr), Y, wsg.ws, 0, q);
    }
    else
    {
        int opad[2] = {int(output_padding[0]), int(output_padding[1])};
        dlprim::Shape rs =
            dlprim::core::Conv2DBase::get_output_shape_transposed(cfg, X.shape(), opad);

        std::swap(cfg.channels_in, cfg.channels_out);
        auto conv = dlprim::core::Conv2DBackwardData::create(ctx, cfg);
        WSGuard wsg(conv->workspace(), input.device());
        result = new_tensor_as(rs, input);
        dlprim::Tensor Y = todp(result);
        conv->enqueue(Y, W, X, wsg.ws, 0, q);
        if (with_bias)
            dlprim::core::add_bias(Y, B, q);
    }
    sync_if_needed(input.device());

    return result;
}

} /* namespace op_plugin */
} /* namespace at_torch */
