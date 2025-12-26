#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

std::tuple<at::Tensor, at::Tensor, at::Tensor> _transform_bias_rescale_qkv(
    const at::Tensor& qkv, const at::Tensor& qkv_bias, const int64_t num_head)
{

    auto qkv_ = qkv.is_nested() ? c10::MaybeOwned<at::Tensor>::owned(qkv.to_padded_tensor(0))
                                : c10::MaybeOwned<at::Tensor>::borrowed(qkv);
    auto B = qkv_->size(0);
    auto T = qkv_->size(1);
    auto _3D = qkv_->size(2);
    auto D = _3D / 3;
    TORCH_CHECK(D % num_head == 0);
    TORCH_CHECK(_3D % 3 == 0);
    const auto dim_per_head = D / num_head;

    const auto qkv_contig = qkv_->expect_contiguous();
    const auto qkv_bias_contig = qkv_bias.expect_contiguous();

    // QKV[5, 7, 66] @ [66]->[3, 5, 2, 7, 11]
    //      B T  3*D   3*dph*3      3  B  nh T dph

    // auto qkv_ in  qkv_contig = ({B, T, 3, num_head, dim_per_head});
    // auto bias_in                     ({3, num_head, dim_per_head});
    auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_->options());
    auto q_k_v_same_order = at::empty({B, T, 3, num_head, dim_per_head}, qkv_->options());
    // this is how to transpose the result to QKV (see below)
    //            B,T,3,num_head,dim_per_head
    /// {1,2}  -> B,3,T,num_head,dim_per_head
    //  {2,3}  -> B,3,num_head,T,dim_per_head
    //  {0,1} ->  3,B,num_head,T,dim_per_head}

    dlprim::Tensor dp_qkv = todp(*qkv_contig);
    dlprim::Tensor dp_bias = todp(*qkv_bias_contig);
    dlprim::Tensor dp_out = todp(q_k_v_same_order);
    dp_qkv.reshape(dlprim::Shape(B * T, _3D));
    dp_out.reshape(dlprim::Shape(B * T, _3D));
    dp_bias.reshape(dlprim::Shape(_3D));
    double scale = 1.0 / std::sqrt(double(dim_per_head));
    dlprim::core::pointwise_operation_broadcast(
        {dp_qkv, dp_bias},
        {dp_out},
        {scale, double(D)},
        {dp_qkv.dtype(), dlprim::int64_data},
        R"xxx(
                    long position_d1 = index.s[1];
                    typeof_x0 scale = position_d1 < w1 ? w0 : 1;
                    y0 = (x0 + x1)*scale;
                )xxx",
        getExecutionContext(qkv),
        false); // don't shrink broadcast to make sure we have correct dims
    at::Tensor q_k_v_cont_as_q_k_v = q_k_v_same_order;
    q_k_v_cont_as_q_k_v = torch::transpose(q_k_v_cont_as_q_k_v, 1, 2);
    q_k_v_cont_as_q_k_v = torch::transpose(q_k_v_cont_as_q_k_v, 2, 3);
    q_k_v_cont_as_q_k_v = torch::transpose(q_k_v_cont_as_q_k_v, 0, 1);
    q_k_v.copy_(q_k_v_cont_as_q_k_v);
    auto q_k_v_s = at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(q_k_v_s.size() == 3);
    return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

} /* namespace op_plugin */
} /* namespace at_torch */
