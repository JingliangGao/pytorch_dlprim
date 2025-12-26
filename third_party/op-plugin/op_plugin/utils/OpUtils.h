#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <iostream>
#include <torch/torch.h>

#include <dlprim/core/activation.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/interpolate.hpp>
#include <dlprim/core/ip.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/pool.hpp>
#include <dlprim/core/util.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/random.hpp>

#include "CLTensor.h"
#include "Utils.h"

namespace at_torch
{
namespace op_plugin
{
struct SeqState
{
    dlprim::RandomState::seed_type seed;
    dlprim::RandomState::sequence_type sequence;
};

SeqState
get_random_seq(c10::Device const& d, int64_t items, c10::optional<at::Generator> generator);

bool is_integral_type(at::Tensor const& t, bool include_bool);
bool is_cpu_scalar(at::Tensor const& other, double& value);
bool IsCPUScalar(const at::Tensor& tensor);

std::pair<dlprim::Shape, dlprim::Shape>
squeeze_dim(dlprim::Shape s, at::OptionalIntArrayRef odim, bool keepdim);
c10::Device ensure_has_index(c10::Device device);

at::Tensor make_contiguous_as_target_type(at::Tensor const& self, at::Tensor const& dst);
at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

} /* namespace op_plugin */
} /* namespace at_torch */
