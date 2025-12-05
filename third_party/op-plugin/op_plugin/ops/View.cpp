#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor view(const at::Tensor & self, at::IntArrayRef size)
    {

        //auto size = C10_AS_at::IntArrayRef_SLOW(sym_size);
        auto inferred_size = at::infer_size_dv(size, self.numel());
        auto stride =
            at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
        TORCH_CHECK(
          stride.has_value(),
          "view size is "
          "not compatible with input tensor's size and stride (at least one dimension"
          " spans across two contiguous subspaces). Use .reshape(...) instead.");


        auto stride_value = *stride;
        at::Tensor data = at::alias(self);
        data.getIntrusivePtr()->set_sizes_and_strides(inferred_size,stride_value);
        return data;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
