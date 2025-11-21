#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    void fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
    {
      TORCH_WARN("The operator '", op.schema().operator_name(), "' is not currently ",
                 "supported on the ocl backend.");
      native::cpu_fallback(op, stack);
    }

}  /* namespace op_plugin */
}  /* namespace at_torch */