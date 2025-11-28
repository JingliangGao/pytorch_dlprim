#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

namespace {
    at::Tensor& cat_internal(const ITensorListRef& tensors, int64_t dim, at::Tensor& out, bool reuse)
    {
        GUARD;

        std::vector<dlprim::Tensor> list;
        std::vector<at::Tensor> list_c;

        // iterate ITensorListRef normally
        for (const at::Tensor& t : tensors) {
            list_c.push_back(t.contiguous());
            list.push_back(todp(list_c.back()));
        }

        TORCH_CHECK(!list_c.empty());
        at::Tensor &ref_tensor = list_c.front();

        size_t total_shape = 0;
        dlprim::Shape ref;

        // shape checks
        for (size_t i = 0; i < list.size(); i++) {
            TORCH_CHECK(0 <= dim && dim < list[i].shape().size(),
                        "dim does not match shape");

            if (i == 0) {
                ref = list[i].shape();
            } else {
                dlprim::Shape s1 = ref, s2 = list[i].shape();
                s1[dim] = 1;
                s2[dim] = 1;
                TORCH_CHECK(s1 == s2, "Shapes do not match");
            }
            total_shape += list[i].shape()[dim];
        }

        ref[dim] = total_shape;

        dlprim::Tensor Y;
        at::Tensor out_c;

        if (reuse) {
            out_c = out.contiguous();
            Y = todp(out_c);
            TORCH_CHECK(Y.shape() == ref, "Output shape incorrect for concat");
        } else {
            out = new_tensor_as(ref, ref_tensor);
            Y = todp(out);
        }

        dlprim::ExecutionContext q(getExecutionContext(ref_tensor));
        dlprim::Context ctx(q);

        dlprim::core::SliceCopy cp(ctx, todp(out.dtype()));

        size_t pos = 0;
        for (size_t i = 0; i < list.size(); i++) {
            at::Tensor new_tensor;
            dlprim::Tensor x;

            if (list_c[i].dtype() != out.dtype()) {
                new_tensor = list_c[i].to(out.dtype());
                x = todp(new_tensor);
            } else {
                x = list[i];
            }

            size_t slice = list[i].shape()[dim];
            cp.tensor_slice_copy(dim, slice,
                                 Y, pos,
                                 x, 0,
                                 0.0, q);
            pos += slice;
        }

        if (reuse && !out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(ref_tensor.device());

        return out;
    }
}



    at::Tensor & cat_out(const ITensorListRef & tensors, int64_t dim, at::Tensor & out)
    {
        GUARD;
        cat_internal(tensors, dim, out, true);
		return out;
    }

    at::Tensor& cat_out(TensorList tensors, Dimname dim, at::Tensor& result)
    {
        TORCH_CHECK(tensors.size() > 0, "cat inputs should not be empty." );
        return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
    }

    at::Tensor _cat(TensorList tensors, int64_t dim)
    {
        GUARD;
        at::Tensor out;
        cat_internal(tensors,dim,out,false);
        return out;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
