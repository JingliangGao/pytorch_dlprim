#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor & amin_out(const at::Tensor & self, IntArrayRef dim, bool keepdim, at::Tensor & out)
    {

        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();

        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor Yval = todp(out_c);
        std::vector<int64_t> dims;
        for(int64_t d :dim) {
            dims.push_back(d);
        }
        if(dims.empty()) {
            for(int i=0;i<X.shape().size();i++)
                dims.push_back(i);
        }
        c10::IntArrayRef sqdims(dims.data(),dims.size());
        auto r = squeeze_dim(X.shape(),sqdims,keepdim);
        TORCH_CHECK(r.second == Yval.shape(),"Invalid output shape");
        Yval.reshape(r.first);

        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        std::string ext_val = dlprim::data_type_to_opencl_numeric_limit(X.dtype(), dlprim::dt_max_val);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                    ctx,
                    {X.specs()},{Yval.specs()},
                    0,X.dtype(),
                    "y0=x0;",
                    "reduce_y0 = " + ext_val + ";",
                    std::string("reduce_y0 = ") + "min" + "(reduce_y0,y0);"
                    );
        WSGuard ws_guard(op->workspace(),self.device());
        op->enqueue({X},{Yval},ws_guard.ws,{},{1,1},{0,0},q);

        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;

    }

    at::Tensor min(const at::Tensor & self)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        at::Tensor result = new_tensor_as(dlprim::Shape(),self);
        dlprim::Tensor Y = todp(result);
        std::string y0 = dlprim::data_type_to_opencl_numeric_limit(X.dtype(), dlprim::dt_max_val);
        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                    ctx,
                    {X.specs()},{Y.specs()},
                    0,X.dtype(),
                    "y0=x0;",
                    std::string("reduce_y0 = ") + y0 + ";",
                    std::string("reduce_y0 = y0 ") + "<" +  " reduce_y0 ? y0 : reduce_y0;"
                    );
        WSGuard ws_guard(op->workspace(),self.device());
        op->enqueue({X},{Y},ws_guard.ws,{},{1},{0},q);

        if (!self.is_contiguous())
            self.copy_(self_c);

        sync_if_needed(self.device());
        return result;
    }


    at::Tensor & minimum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c  = self.contiguous();
        at::Tensor out_c = out.contiguous();
        at::Tensor other_c = other.contiguous();

        std::string op_builder = "y0 = min(left,right); ";

        dlprim::Tensor y(todp(out_c));
        double value;
        if(is_cpu_scalar(other,value)) {
            dlprim::Tensor x0(todp(self_c));
            dlprim::core::pointwise_operation_broadcast({x0},{y},{value},{x0.dtype()},
                        "typeof_x0 left = x0; typeof_w0 right = w0;" + op_builder,
                        getExecutionContext(self));
            sync_if_needed(self.device());
        }
        else if(is_cpu_scalar(self,value)) {
            dlprim::Tensor x0(todp(other_c));
            dlprim::core::pointwise_operation_broadcast({x0},{y},{value},{x0.dtype()},
                        "typeof_w0 left = w0; typeof_x0 right = x0;" + op_builder,
                        getExecutionContext(other));
            sync_if_needed(other.device());
        }
        else {
            dlprim::Tensor x0(todp(self_c));
            dlprim::Tensor x1(todp(other_c));
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y},{},
                    "typeof_x0 left = x0; typeof_x1 right = x1;" + op_builder,
                    getExecutionContext(self));
            sync_if_needed(self.device());
        }

        if (!out.is_contiguous())
            out.copy_(out_c);

        return out;
    }

    }  /* namespace op_plugin */
}  /* namespace at_torch */
