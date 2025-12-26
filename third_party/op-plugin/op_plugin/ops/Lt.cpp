#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

at::Tensor& lt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{

    at::Tensor self_c = self.contiguous();
    at::Tensor out_c = out.contiguous();
    at::Tensor other_c = other.contiguous();

    std::string op_builder = "y0 = (left < right);";

    dlprim::Tensor y(todp(out_c));
    double value;
    if (is_cpu_scalar(other, value))
    {
        dlprim::Tensor x0(todp(self_c));
        dlprim::core::pointwise_operation_broadcast(
            {x0},
            {y},
            {value},
            {x0.dtype()},
            "typeof_x0 left = x0; typeof_w0 right = w0;" + op_builder,
            getExecutionContext(self));
        sync_if_needed(self.device());
    }
    else if (is_cpu_scalar(self, value))
    {
        dlprim::Tensor x0(todp(other_c));
        dlprim::core::pointwise_operation_broadcast(
            {x0},
            {y},
            {value},
            {x0.dtype()},
            "typeof_w0 left = w0; typeof_x0 right = x0;" + op_builder,
            getExecutionContext(other));
        sync_if_needed(other.device());
    }
    else
    {
        dlprim::Tensor x0(todp(self_c));
        dlprim::Tensor x1(todp(other_c));
        dlprim::core::pointwise_operation_broadcast(
            {x0, x1},
            {y},
            {},
            "typeof_x0 left = x0; typeof_x1 right = x1;" + op_builder,
            getExecutionContext(self));
        sync_if_needed(self.device());
    }

    if (!out.is_contiguous())
        out.copy_(out_c);

    return out;
}

at::Tensor& lt_out(const at::Tensor& self, const Scalar& other, at::Tensor& out)
{

    at::Tensor self_c = self.contiguous();
    at::Tensor out_c = out.contiguous();

    dlprim::Tensor x0 = todp(self_c);
    dlprim::Tensor y0 = todp(out_c);
    float w0 = other.toDouble();
    dlprim::core::pointwise_operation_broadcast(
        {x0},
        {y0},
        {w0},
        {dlprim::float_data},
        "y0 = (x0 < w0) ? 1 : 0;",
        getExecutionContext(self));

    if (!out.is_contiguous())
        out.copy_(out_c);

    sync_if_needed(self.device());
    return out;
}

} /* namespace op_plugin */
} /* namespace at_torch */
