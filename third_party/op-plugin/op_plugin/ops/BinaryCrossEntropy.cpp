#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction)
{

    TORCH_CHECK(!weight || weight->numel() == 0, "Weight in binar_cross_entroy isn't supported");
    at::Tensor self_c = self.contiguous();
    at::Tensor target_c = target.contiguous();
    dlprim::Tensor x = todp(self_c);
    dlprim::Tensor y = todp(target_c);
    bool reduce = false;
    double scale = 1;
    switch (reduction)
    {
    case 0:
        reduce = false;
        break; // None
    case 1:
        reduce = true;
        scale = 1.0 / x.shape().total_size();
        break; // Mean
    case 2:
        reduce = true;
        break; // sum
    }
    dlprim::Shape target_shape;
    if (!reduce)
        target_shape = x.shape();
    at::Tensor loss_tensor = new_tensor_as(target_shape, self_c);
    dlprim::Tensor loss(todp(loss_tensor));
    auto q = getExecutionContext(self);
    dlprim::Context ctx(q);
    auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
        ctx,
        {x.specs(), y.specs()},
        {loss.specs()},
        0,
        dlprim::float_data,
        "y0 = - (x1 * max((typeof_x0)(-100),log(x0)) + (1-x1) * "
        "max((typeof_x0)(-100),log(1-x0)));",
        "reduce_y0 = 0;",
        "reduce_y0 += y0;");
    WSGuard wsg(op->workspace(), self.device());
    op->enqueue({x, y}, {loss}, wsg.ws, {}, {scale}, {0}, q);
    sync_if_needed(self.device());
    return loss_tensor;
}

} /* namespace op_plugin */
} /* namespace at_torch */
