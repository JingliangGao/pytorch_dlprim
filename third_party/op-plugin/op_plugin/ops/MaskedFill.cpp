#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {

        at::Tensor self_c = self.contiguous();
        at::Tensor mask_c = mask.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor M = todp(mask_c);
        TORCH_CHECK(X.shape() == M.shape(), "Broadasting is not implemented in masked_fill yet");
        auto ec = getExecutionContext(self);
        X.to_host(ec);
        M.to_host(ec);

        double v = value.to<double>();

        auto fill_by_mask = [&](auto *mask_ptr){
            size_t N = X.shape().total_size();
            switch(M.dtype()){
            default: break;
            }
            switch(X.dtype()){
            case dlprim::float_data: {
                float *px = X.data<float>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<float>(v);
                break; }
            case dlprim::double_data: {
                double *px = X.data<double>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<double>(v);
                break; }
            case dlprim::int8_data: {
                int8_t *px = X.data<int8_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<int8_t>(v);
                break; }
            case dlprim::uint8_data: {
                uint8_t *px = X.data<uint8_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<uint8_t>(v);
                break; }
            case dlprim::int16_data: {
                int16_t *px = X.data<int16_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<int16_t>(v);
                break; }
            case dlprim::uint16_data: {
                uint16_t *px = X.data<uint16_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<uint16_t>(v);
                break; }
            case dlprim::int32_data: {
                int32_t *px = X.data<int32_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<int32_t>(v);
                break; }
            case dlprim::uint32_data: {
                uint32_t *px = X.data<uint32_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<uint32_t>(v);
                break; }
            case dlprim::int64_data: {
                int64_t *px = X.data<int64_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<int64_t>(v);
                break; }
            case dlprim::uint64_data: {
                uint64_t *px = X.data<uint64_t>();
                for(size_t i=0;i<N;i++) if(mask_ptr[i]) px[i] = static_cast<uint64_t>(v);
                break; }
            default:
                TORCH_CHECK(!"Not implemented dtype","Not implemented");
            }
        };

        switch(M.dtype()){
        case dlprim::float_data:
            fill_by_mask(M.data<float>());
            break;
        case dlprim::double_data:
            fill_by_mask(M.data<double>());
            break;
        case dlprim::int8_data:
            fill_by_mask(M.data<int8_t>());
            break;
        case dlprim::uint8_data:
            fill_by_mask(M.data<uint8_t>());
            break;
        case dlprim::int16_data:
            fill_by_mask(M.data<int16_t>());
            break;
        case dlprim::uint16_data:
            fill_by_mask(M.data<uint16_t>());
            break;
        case dlprim::int32_data:
            fill_by_mask(M.data<int32_t>());
            break;
        case dlprim::uint32_data:
            fill_by_mask(M.data<uint32_t>());
            break;
        case dlprim::int64_data:
            fill_by_mask(M.data<int64_t>());
            break;
        case dlprim::uint64_data:
            fill_by_mask(M.data<uint64_t>());
            break;
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }

        // push modified data back to device
        X.to_device(ec,X.host_data());

        if (!self.is_contiguous())
            self.copy_(self_c);

        sync_if_needed(self.device());
        return self;
    }

    at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value){
        // bring value to CPU and extract scalar
        at::Tensor val_cpu = value.cpu();
        TORCH_CHECK(val_cpu.dim() == 0, "masked_fill only supports a 0-dimensional value tensor");
        return at_torch::op_plugin::masked_fill_(self, mask, val_cpu.item());
    }



  }  /* namespace op_plugin */
}  /* namespace at_torch */
