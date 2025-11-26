#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/InferSize.h>
#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <iostream>
#include <torch/torch.h>
#include "OpInterface.h"

void fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
  TORCH_WARN("The operator '", op.schema().operator_name(), "' is not currently ",
             "supported on the PrivateUse1 backend.");
  at::native::cpu_fallback(op, stack);
}

namespace at_torch {
namespace ${namespace} {

    ${declarations}
    
}
}  /* namespace at_torch */ 

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
      m.fallback(torch::CppFunction::makeFromBoxedFunction<&fallback>());
}

TORCH_LIBRARY_IMPL(${p_namespace}, PrivateUse1, m) {
    ${p_declarations}
}

TORCH_LIBRARY_IMPL(${ap_namespace}, AutogradPrivateUse1, m) {
    ${ap_declarations}
} 