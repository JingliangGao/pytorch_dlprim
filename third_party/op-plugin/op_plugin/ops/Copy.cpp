#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
        GUARD;

        // If same storage / same tensor, nothing to do
        if (self.unsafeGetTensorImpl() == src.unsafeGetTensorImpl()) {
            return self;
        }

        // If both are zero-sized, nothing to do
        if (self.numel() == 0 && src.numel() == 0) {
            return self;
        }

        // Reuse existing implementation that copies src -> dst
        // Note: _copy_from defined earlier has signature (src, dst, non_blocking)
        at_torch::op_plugin::_copy_from(src, self, non_blocking);

        return self;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
