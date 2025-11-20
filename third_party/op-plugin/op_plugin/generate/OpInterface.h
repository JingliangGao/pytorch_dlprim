#include "CLTensor.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/InferSize.h>
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


namespace at_torch {
namespace op_plugin {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;
using torch::Tensor;
using c10::Device;
using c10::DeviceType;

::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight);
at::Tensor & nll_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input);
}  /* namespace op_plugin */
}  /* namespace at_torch */
