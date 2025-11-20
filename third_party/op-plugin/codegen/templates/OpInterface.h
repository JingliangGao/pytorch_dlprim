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
namespace ${namespace} {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;
using torch::Tensor;
using c10::Device;
using c10::DeviceType;

${declarations}
}  /* namespace ${namespace} */
}  /* namespace at_torch */
