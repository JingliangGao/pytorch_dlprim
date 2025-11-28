#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    template<typename E,typename M>
    size_t select_impl_by(E *p,M *m,size_t n)
    {
        size_t N = 0;
        for(size_t i=0;i<n;i++) {
            if(m[i]) {
                p[N] = p[i];
                N++;
            }
        }
        return N;
    }

    template<typename T>
    size_t select_impl(T *mask,dlprim::Tensor &/*m*/,dlprim::Tensor &v)
    {
        void *p=v.host_data();
        switch(dlprim::size_of_data_type(v.dtype())) {
        case 1: return select_impl_by(static_cast<int8_t  *>(p),mask,v.shape().total_size());
        case 2: return select_impl_by(static_cast<int16_t *>(p),mask,v.shape().total_size());
        case 4: return select_impl_by(static_cast<int32_t *>(p),mask,v.shape().total_size());
        case 8: return select_impl_by(static_cast<int64_t *>(p),mask,v.shape().total_size());
        default:
            TORCH_CHECK(!"Invalid sizeof");
            return 0;
        }
    }


    at::Tensor masked_select(const at::Tensor & self, const at::Tensor & mask)
    {
        GUARD;
        at::Tensor self_c = self.contiguous();
        at::Tensor mask_c = mask.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor m = todp(mask_c);
        TORCH_CHECK(x.shape() == m.shape(),"Broadasting is not implemented in masked_select yet");
        auto ec = getExecutionContext(self);
        x.to_host(ec);
        m.to_host(ec);
        size_t N = 0;
        switch(m.dtype()) {
        case dlprim::float_data:
            N = select_impl(m.data<float>(),m,x);
            break;
        case dlprim::double_data:
            N = select_impl(m.data<double>(),m,x);
            break;
        case dlprim::int8_data:
            N = select_impl(m.data<int8_t>(),m,x);
            break;
        case dlprim::uint8_data:
            N = select_impl(m.data<uint8_t>(),m,x);
            break;
        case dlprim::int16_data:
            N = select_impl(m.data<int16_t>(),m,x);
            break;
        case dlprim::uint16_data:
            N = select_impl(m.data<uint16_t>(),m,x);
            break;
        case dlprim::int32_data:
            N = select_impl(m.data<int32_t>(),m,x);
            break;
        case dlprim::uint32_data:
            N = select_impl(m.data<uint32_t>(),m,x);
            break;
        case dlprim::int64_data:
            N = select_impl(m.data<int64_t>(),m,x);
            break;
        case dlprim::uint64_data:
            N = select_impl(m.data<uint64_t>(),m,x);
            break;
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }
        at::Tensor res=new_tensor_as(dlprim::Shape(N),self);
        if(N > 0) {
            dlprim::Tensor y=todp(res);
            y.to_device(getExecutionContext(self),x.host_data());
        }
        sync_if_needed(self.device());
        return res;
    }

  }  /* namespace op_plugin */
}  /* namespace at_torch */
