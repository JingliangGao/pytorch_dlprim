#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    ::std::tuple<at::Tensor,at::Tensor> native_dropout(const at::Tensor & input, double p, ::std::optional<bool> train)
    {

        at::Tensor input_c = input.contiguous();
        dlprim::Tensor X = todp(input_c);
        at::Tensor mask = new_tensor_as(X.shape(),input_c);
        at::Tensor res =  new_tensor_as(X.shape(),input_c);
        dlprim::Tensor Y = todp(res);
        dlprim::Tensor M = todp(mask);
        if(train && *train && p > 0) {
            bernoulli_(mask,1-p,c10::nullopt);
            dlprim::core::pointwise_operation({X,M},{Y},{1/(1-p)},
                                          "y0 = x0*x1*w0;",
                                          getExecutionContext(input));
        }
        else {
            torch::fill_(mask,1);
        }
        sync_if_needed(input.device());
        return std::make_pair(res,mask);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
