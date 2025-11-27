#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

namespace {     
    at::Tensor & binary_op_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out,std::string const &op,std::string op_builder="")
    {
        GUARD;
        at::Tensor self_c  = self.contiguous();
        at::Tensor out_c = out.contiguous();
        at::Tensor other_c = other.contiguous();
        
        if(op_builder.empty()) {
            op_builder = "y0 = left " + op + " right;";
        }

        dlprim::Tensor y(todp(out_c));
        double value;
        if(is_cpu_scalar(other, value)) {
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
}

    at::Tensor & addcmul_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const Scalar & value, at::Tensor & out)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor out_c = out.contiguous();
        at::Tensor tensor1_c = tensor1.contiguous();
        at::Tensor tensor2_c = tensor2.contiguous();
        
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor x1=todp(tensor1_c);
        dlprim::Tensor x2=todp(tensor2_c);
        dlprim::Tensor y0=todp(out_c);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation_broadcast({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * x1 * x2;",
                                      getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */