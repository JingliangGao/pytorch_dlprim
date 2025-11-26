#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    static at::Tensor & log_softmax_out_nocheck(const at::Tensor & self, int64_t dim, bool is_log, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out);
        TORCH_CHECK(dim==-1 || (0<=dim && dim < int(x.shape().size())),"Invalid value of dim");
        if(x.shape().size()!=2) {
            if(dim == -1)
                dim = x.shape().size() - 1;
            int N=1,M=1;
            int Rd = x.shape()[dim];
            for(int i=0;i<dim;i++)
                N*=x.shape()[i];
            for(int i=dim+1;i<int(x.shape().size());i++)
                M*=x.shape()[i];
            auto new_shape = dlprim::Shape(N,Rd,M);
            x.reshape(new_shape);
            y.reshape(new_shape);
        }
        dlprim::core::softmax_forward(x,y,is_log,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    at::Tensor & _log_softmax_out(const at::Tensor & self, int64_t dim, bool /*half_to_float*/, at::Tensor & out)
    {
        return log_softmax_out_nocheck(self,dim,true,out);
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */