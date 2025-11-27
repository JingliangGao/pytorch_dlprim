#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    
    at::Tensor & argmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();
        
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor Yind = todp(out_c);
        std::vector<int64_t> dims;
        if(dim) {
            dims.push_back(*dim);
        }
        else {
            for(int i=0;i<X.shape().size();i++)
                dims.push_back(i);
        }
        c10::IntArrayRef sqdims(dims.data(),dims.size());
        auto r = squeeze_dim(X.shape(),sqdims,keepdim);
        TORCH_CHECK(r.second == Yind.shape(),"Invalid output shape");
        Yind.reshape(r.first);

        WSGuard tmp_guard(Yind.shape().total_size()*dlprim::size_of_data_type(X.dtype()),
                         self.device());
        dlprim::Tensor Yval = tmp_guard.ws.sub_tensor(0,Yind.shape(),X.dtype());

        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        std::string min_val = dlprim::data_type_to_opencl_numeric_limit(X.dtype(),dlprim::dt_min_val);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                    ctx,
                    {X.specs()},{Yval.specs(),Yind.specs()},
                    0,dlprim::float_data,
                    "y0=x0; y1=reduce_item;",
                    "reduce_y0 = " + min_val + "; reduce_y1 = -1;",
                    R"xxx(
                        if(y0 > reduce_y0) {
                            reduce_y0 = y0; 
                            reduce_y1 = y1; 
                        }
                    )xxx"
                    );
        WSGuard ws_guard(op->workspace(),self.device());
        op->enqueue({X},{Yval,Yind},ws_guard.ws,{},{1,1},{0,0},q);
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }
    

    }  /* namespace op_plugin */
}  /* namespace at_torch */