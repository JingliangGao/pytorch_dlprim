#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


   at::Tensor &fill_(at::Tensor &self, const c10::Scalar &value)
    {

        dlprim::Tensor t(todp(self));
        auto q = getExecutionContext(self);
        dlprim::core::fill_tensor(t,value.to<double>(),q);
        sync_if_needed(self.device());
        return self;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
