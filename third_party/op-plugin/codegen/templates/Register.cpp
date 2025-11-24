#include <torch/library.h>
#include <ATen/ATen.h>
#include "OpInterface.h"

namespace at_torch {
namespace ${namespace} {

    ${declarations}
    
}
}  /* namespace at_torch */ 

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
      m.fallback(torch::CppFunction::makeFromBoxedFunction<&at_torch::op_plugin::fallback>());
}

TORCH_LIBRARY_IMPL(${p_namespace}, PrivateUse1, m) {
    ${p_declarations}
}

TORCH_LIBRARY_IMPL(${ap_namespace}, AutogradPrivateUse1, m) {
    ${ap_declarations}
} 