#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    ::std::tuple<at::Tensor &,at::Tensor &> _native_multi_head_attention_out(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type, at::Tensor & out0, at::Tensor & out1)
    {

        auto r = at::cpu::_native_multi_head_attention(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,need_weights,average_attn_weights,mask_type);
        out0.copy_(std::get<0>(r));
        out1.copy_(std::get<1>(r));
        return ::std::tuple<at::Tensor &,at::Tensor &>(out0,out1);
    }


    ::std::tuple<at::Tensor,at::Tensor> _native_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type)
    {

        return at::cpu::_native_multi_head_attention(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,need_weights,average_attn_weights,mask_type);
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
