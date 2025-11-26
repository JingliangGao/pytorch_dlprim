#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    bool _has_compatible_shallow_copy_type(const at::Tensor & self, const at::Tensor & from) {
        c10::DispatchKeySet self_keyset = self.key_set();
        c10::DispatchKeySet from_keyset = from.key_set();
        auto is_dense = [](c10::DispatchKeySet ks) {
            return ks.has(c10::DispatchKey::CPU) || ks.has(c10::DispatchKey::PrivateUse1);
        };
        return (self_keyset == from_keyset) || (is_dense(self_keyset) && is_dense(from_keyset));
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */