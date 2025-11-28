#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor _copy_from_and_resize(const at::Tensor & self, const at::Tensor & dst)
    {
        return at_torch::op_plugin::_copy_from(self,dst,false);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
