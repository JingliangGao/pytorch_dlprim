#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {



    ::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_out(const at::Tensor & self, at::Tensor & output, at::Tensor & buffer)
    {

        at::Tensor self_c = self.contiguous(), output_c = output.contiguous();
        at::Tensor buffer_c = buffer.contiguous();
        dlprim::Tensor x=todp(self_c), out = todp(output_c), buf = todp(buffer_c);
        dlprim::core::pointwise_operation({x},{out,buf},{},
                    R"xxx(
                    y1 = exp(-fabs(x0));
                    y0 = min((dtype)(0),x0) - log1p(y1);
                    )xxx",
                    getExecutionContext(self));

        if(!output.is_contiguous())
            output.copy_(output_c);

        if(!buffer.is_contiguous())
            buffer.copy_(buffer_c);

        sync_if_needed(self.device());
        return ::std::tuple<at::Tensor &,at::Tensor &>(output,buffer);

    }


    ::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(const at::Tensor & self)
    {

        at::Tensor out = at::empty_like(self);
        at::Tensor buffer = at::empty_like(self);

        log_sigmoid_forward_out(self,out,buffer);
        return ::std::tuple<at::Tensor,at::Tensor>(out,buffer);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
