#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& log_softmax_backward_data_out_nocheck(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    bool is_log,
    at::Tensor& out)
{

    dlprim::Tensor dx = todp(out);
    at::Tensor output_c = output.contiguous(), grad_output_c = grad_output.contiguous();
    dlprim::Tensor y = todp(output_c);
    dlprim::Tensor dy = todp(grad_output_c);
    TORCH_CHECK(dim == -1 || (0 <= dim && dim < int(y.shape().size())), "Invalid value of dim");
    if (y.shape().size() != 2)
    {
        if (dim == -1)
            dim = y.shape().size() - 1;
        int N = 1, M = 1;
        int Rd = y.shape()[dim];
        for (int i = 0; i < dim; i++)
            N *= y.shape()[i];
        for (int i = dim + 1; i < int(y.shape().size()); i++)
            M *= y.shape()[i];
        auto new_shape = dlprim::Shape(N, Rd, M);
        dx.reshape(new_shape);
        dy.reshape(new_shape);
        y.reshape(new_shape);
    }

    dlprim::core::softmax_backward(dx, y, dy, is_log, 0.0f, getExecutionContext(grad_output));
    sync_if_needed(grad_output.device());
    return out;
}

at::Tensor& _log_softmax_backward_data_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    ScalarType /*input_dtype*/,
    at::Tensor& out)
{
    return log_softmax_backward_data_out_nocheck(grad_output, output, dim, true, out);
}

} /* namespace op_plugin */
} /* namespace at_torch */
