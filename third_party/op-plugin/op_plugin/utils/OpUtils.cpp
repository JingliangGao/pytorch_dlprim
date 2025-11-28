#include "OpUtils.h"

namespace at_torch {
namespace op_plugin {

    bool is_integral_type(at::Tensor const &t,bool include_bool)
    {
        return c10::isIntegralType(t.dtype().toScalarType(),include_bool);
    }

    bool is_cpu_scalar(at::Tensor const &other, double &value)
    {
        if(other.device() == c10::Device(c10::kCPU) && other.numel()==1) {
            switch(other.dtype().toScalarType()){
            case c10::kFloat:
                value = *static_cast<float const *>(other.data_ptr());
                break;
            case c10::kDouble:
                value = *static_cast<double const *>(other.data_ptr());
                break;
            case c10::kLong:
                value = *static_cast<int64_t const *>(other.data_ptr());
                break;
            default:
                TORCH_CHECK(false,"Unsupported cpu data type");
            }
            return true;
        }
        return false;
    }


    std::pair<dlprim::Shape, dlprim::Shape> squeeze_dim(dlprim::Shape s, at::OptionalIntArrayRef odim, bool keepdim)
    {
        GUARD;
        std::vector<size_t> full,squeezed;
        std::vector<int> dims;
        if(odim)
            dims.assign(odim->begin(),odim->end());
        if(dims.empty()) {
            for(int i=0;i<s.size();i++)
                dims.push_back(i);
        }

        for(auto &axis : dims) {
            if (axis < 0) {
                axis = axis + s.size();
            }
        }
        std::sort(dims.begin(),dims.end());
        int pos = 0;
        for(int i=0;i<s.size();i++) {
            if(pos < int(dims.size()) && i==dims[pos]) {
                full.push_back(1);
                if(keepdim)
                    squeezed.push_back(1);
                pos++;
            }
            else {
                full.push_back(s[i]);
                squeezed.push_back(s[i]);
            }
        }
        TORCH_CHECK(pos == int(dims.size()),"Looks like invalid dims");
        auto full_shape = dlprim::Shape::from_range(full.begin(),full.end());
        auto squeezed_shape = dlprim::Shape::from_range(squeezed.begin(),squeezed.end());
        if(squeezed_shape.size() == 0) {
            squeezed_shape = dlprim::Shape(1);
        }
        return std::make_pair(full_shape,squeezed_shape);
    }

    SeqState get_random_seq(c10::Device const &d,int64_t items,c10::optional<at::Generator> generator)
    {
        dlprim::RandomState &state = CLContextManager::instance().rng_state(d.index());
        size_t rounds = (items +  dlprim::philox::result_items - 1) / dlprim::philox::result_items;
        SeqState s;
        s.seed = state.seed();
        s.sequence  = state.sequence_bump(rounds);
        return  s;
    }

    c10::Device ensure_has_index(c10::Device device) {

        if (device.is_cpu() || device.has_index()) {
            return device;
        }

        if (device.type() == OpenCLDeviceType) {
            return c10::Device(OpenCLDeviceType, 0);
        }

        const c10::impl::DeviceGuardImplInterface* impl =
            c10::impl::getDeviceGuardImpl(device.type());
        return impl->getDevice();
    }

    at::Tensor make_contiguous_as_target_type(at::Tensor const &self,at::Tensor const &dst)
    {
        GUARD;
        at::Tensor c_src = self;
        if(self.dtype() != dst.dtype() || !self.is_contiguous()) {
            c10::TensorOptions options = c10::TensorOptions().dtype(dst.dtype()).memory_format(c10::MemoryFormat::Contiguous);
            at::Tensor temp = at::empty_like(c_src, options);
            temp.copy_(c_src);
            c_src = temp;
        }
        return c_src;
    }

    at::Tensor _copy_from(const at::Tensor & self, const at::Tensor & dst, bool non_blocking)
    {
        GUARD;
        if(self.numel() == 0 && dst.numel() == 0) {
            return self;
        }
        
        if(dst.device().type() == c10::DeviceType::CPU && self.device().type() == OpenCLDeviceType) {
            at::Tensor c_src = make_contiguous_as_target_type(self,dst);
            dlprim::Tensor t = todp(c_src);
            auto ec = getExecutionContext(self);
            if(dst.is_contiguous()) {
                void *ptr = dst.data_ptr();
                t.to_host(ec,ptr);
            }
            else {
                c10::TensorOptions options = c10::TensorOptions().memory_format(c10::MemoryFormat::Contiguous);
                at::Tensor dst_c = at::empty_like(dst,options);
                void *ptr = dst_c.data_ptr();
                t.to_host(ec,ptr);
                dst.copy_(dst_c);
            }
        }
        else if(self.device().type() == c10::DeviceType::CPU && dst.device().type() == OpenCLDeviceType) {
            at::Tensor c_src = make_contiguous_as_target_type(self,dst);
            auto ec = getExecutionContext(dst);
            if(dst.is_contiguous()) {
                dlprim::Tensor t(todp(dst));
                t.to_device(ec,c_src.data_ptr());
            }
            else {
                c10::TensorOptions options = c10::TensorOptions().memory_format(c10::MemoryFormat::Contiguous);
                at::Tensor temp = at::empty_like(dst,options);
                dlprim::Tensor t(todp(temp));
                t.to_device(ec,c_src.data_ptr());
                dst.copy_(temp);
            }
        }
        else if(self.device().type() == OpenCLDeviceType && dst.device().type() == OpenCLDeviceType) {
            if(self.is_contiguous() && dst.is_contiguous()) {
                dlprim::core::pointwise_operation_broadcast({todp(self)},{todp(dst)},{},"y0=x0;",getExecutionContext(self.device()));
            }
            else {
                auto src_sizes  = self.sizes();
                auto src_stride = self.strides();
                auto src_offset = self.storage_offset();
                auto tgt_sizes  = dst.sizes();
                auto tgt_stride = dst.strides();
                auto tgt_offset = dst.storage_offset();
                TORCH_CHECK(src_sizes == tgt_sizes);
                dlprim::Shape shape=dlprim::Shape::from_range(src_sizes.begin(),src_sizes.end());
                dlprim::Shape src_std=dlprim::Shape::from_range(src_stride.begin(),src_stride.end());
                dlprim::Shape tgt_std=dlprim::Shape::from_range(tgt_stride.begin(),tgt_stride.end());
                dlprim::core::copy_strided(shape,buffer_from_tensor(self),src_offset,src_std,
                                                 buffer_from_tensor(dst), tgt_offset,tgt_std,
                                                 todp(self.dtype()),
                                                 todp(dst.dtype()),
                                                 getExecutionContext(self.device()));
            }
            if(non_blocking)
                sync_if_needed(self.device());
            else
                getExecutionContext(self.device()).queue().flush();
        }
        else {
            throw std::runtime_error("OpenCL supports copy to CPU backend only");
        }
        return self;
    }

}  /* namespace op_plugin */
}  /* namespace at_torch */