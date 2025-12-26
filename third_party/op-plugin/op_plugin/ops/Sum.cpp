#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& sum_out(
    const at::Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    at::Tensor& out)
{

    at::Tensor self_c = self.contiguous();
    at::Tensor out_c = out.contiguous();

    dlprim::Tensor X = todp(self_c);
    auto r = squeeze_dim(X.shape(), dim, keepdim);
    dlprim::Tensor Y = todp(out_c);
    TORCH_CHECK(r.second == Y.shape(), "Invalid output shape");
    Y.reshape(r.first);

    double scale = 1;

    auto q = getExecutionContext(self);
    dlprim::Context ctx(q);
    auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
        ctx,
        {X.specs()},
        {Y.specs()},
        0,
        dlprim::float_data,
        "y0=x0;",
        "reduce_y0 = 0;",
        "reduce_y0 += y0;");

    WSGuard wsg(op->workspace(), self.device());
    op->enqueue({X}, {Y}, wsg.ws, {}, {scale}, {0}, q);

    if (!out.is_contiguous())
        out.copy_(out_c);

    sync_if_needed(self.device());
    return out;
}

} /* namespace op_plugin */
} /* namespace at_torch */
