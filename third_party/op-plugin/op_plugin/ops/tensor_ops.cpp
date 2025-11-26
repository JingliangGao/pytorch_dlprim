#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    static inline c10::Device ensure_has_index(c10::Device device) {
        // 对 OpenCL 后端，若没有 index，则默认回退到 ocl:0
        if (device.is_cpu() || device.has_index()) {
            return device;
        }
        // 如果 DeviceType 是自定义的 OpenCLDeviceType，无法保证 getDeviceGuardImpl 可用，
        // 所以直接返回默认 OpenCL 设备(0)。
        if (device.type() == OpenCLDeviceType) {
            return c10::Device(OpenCLDeviceType, 0);
        }
        // Fallback 使用 PyTorch 提供的 impl（可用于其它 device types）
        const c10::impl::DeviceGuardImplInterface* impl =
            c10::impl::getDeviceGuardImpl(device.type());
        return impl->getDevice();
    }

    // {"schema": "aten::empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor", "dispatch": "True", "default": "False"}
    torch::Tensor empty(IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format)
    {
        GUARD;
        TORCH_CHECK(!layout || *layout == Layout::Strided,"pytorch_ocl supports only strided layout")
        // FIX ME Later -how to handle non Contiguous format {
        //TORCH_CHECK(!memory_format || *memory_format == MemoryFormat::Contiguous,"Contigonous format expected");
        // }
        c10::Device dev = device ? *device : Device(OpenCLDeviceType,0);
        c10::ScalarType st = dtype ? *dtype : c10::kFloat; 
        if(st == c10::kDouble && !CLContextManager::fp64(dev.index())) {
            st = c10::kFloat;
            TORCH_WARN("This device ocl:" + std::to_string(dev.index()) + " does not support cl_khr_fp64, falling back to float");
        }
        return at_torch::new_ocl_tensor(size,dev,st);
    }


    torch::Tensor _reshape_alias(const Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride)
    {
        GUARD;
        torch::Tensor data = at::alias(self);
        data.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        return data;
    }

/// "aten::empty_strided"
    Tensor empty_strided(IntArrayRef size, IntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) 
    {
        GUARD;
        Tensor r = empty(size,dtype,layout,device,pin_memory,c10::nullopt);
        return at_torch::op_plugin::_reshape_alias(r,size,stride);
    }

    Tensor view_old(const Tensor & self, IntArrayRef size)
    {
        GUARD;
        torch::Tensor data=at::alias(self);
        TORCH_CHECK(data.is_contiguous(),"View imlemented on contiguous array");
        std::vector<int64_t> v(size.begin(),size.end());
        int64_t total=1,index=-1;
        for(unsigned i=0;i<v.size();i++) {
            if(v[i] == -1) {
                TORCH_CHECK(index==-1,"Must be unique -1");
                index=i;
            }
            else {
                total *= v[i];
            }
        }
        if(index != -1) {
            TORCH_CHECK(self.numel() % total == 0);
            v[index] = self.numel() / total;
        }
        else {
            TORCH_CHECK(total == self.numel());
        }
        c10::IntArrayRef new_size(v.data(),v.size());
        data.getIntrusivePtr()->set_sizes_contiguous(new_size);
        return data;
    }


    Tensor view(const Tensor & self, c10::IntArrayRef size)
    {
        GUARD;
        //auto size = C10_AS_INTARRAYREF_SLOW(sym_size);
        auto inferred_size = at::infer_size_dv(size, self.numel());
        auto stride =
            at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
        TORCH_CHECK(
          stride.has_value(),
          "view size is "
          "not compatible with input tensor's size and stride (at least one dimension"
          " spans across two contiguous subspaces). Use .reshape(...) instead.");


        auto stride_value = *stride;
        Tensor data = at::alias(self);
        data.getIntrusivePtr()->set_sizes_and_strides(inferred_size,stride_value);
        return data;
    }

    static Tensor make_contiguous_as_target_type(Tensor const &self,Tensor const &dst)
    {
        GUARD;
        Tensor c_src = self;
        if(self.dtype() != dst.dtype() || !self.is_contiguous()) {
            TensorOptions options = TensorOptions().dtype(dst.dtype()).memory_format(MemoryFormat::Contiguous);
            Tensor temp = at::empty_like(c_src,options);
            temp.copy_(c_src);
            c_src = temp;
        }
        return c_src;
    }

    Tensor _copy_from(const Tensor & self, const Tensor & dst, bool non_blocking)
    {
        GUARD;
        if(self.numel() == 0 && dst.numel() == 0) {
            return self;
        }
        
        if(dst.device().type() == c10::DeviceType::CPU && self.device().type() == OpenCLDeviceType) {
            Tensor c_src = make_contiguous_as_target_type(self,dst);
            dlprim::Tensor t = todp(c_src);
            auto ec = getExecutionContext(self);
            if(dst.is_contiguous()) {
                void *ptr = dst.data_ptr();
                t.to_host(ec,ptr);
            }
            else {
                TensorOptions options = TensorOptions().memory_format(MemoryFormat::Contiguous);
                Tensor dst_c = at::empty_like(dst,options);
                void *ptr = dst_c.data_ptr();
                t.to_host(ec,ptr);
                dst.copy_(dst_c);
            }
        }
        else if(self.device().type() == c10::DeviceType::CPU && dst.device().type() == OpenCLDeviceType) {
            Tensor c_src = make_contiguous_as_target_type(self,dst);
            auto ec = getExecutionContext(dst);
            if(dst.is_contiguous()) {
                dlprim::Tensor t(todp(dst));
                t.to_device(ec,c_src.data_ptr());
            }
            else {
                TensorOptions options = TensorOptions().memory_format(MemoryFormat::Contiguous);
                Tensor temp = at::empty_like(dst,options);
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

    // {"schema": "aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor _copy_from_and_resize(const Tensor & self, const Tensor & dst)
    {
        return at_torch::op_plugin::_copy_from(self,dst,false);
    }

    // {"schema": "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking) {
        GUARD;

        // If same storage / same tensor, nothing to do
        if (self.unsafeGetTensorImpl() == src.unsafeGetTensorImpl()) {
            return self;
        }

        // If both are zero-sized, nothing to do
        if (self.numel() == 0 && src.numel() == 0) {
            return self;
        }

        // Reuse existing implementation that copies src -> dst
        // Note: _copy_from defined earlier has signature (src, dst, non_blocking)
        at_torch::op_plugin::_copy_from(src, self, non_blocking);

        return self;
    }

    at::Tensor _to_copy(const Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, ::std::optional<at::MemoryFormat> memory_format){
        GUARD;

        // 1) 构造 override options 并合并
        c10::TensorOptions opts_override;
        if (dtype.has_value()) opts_override = opts_override.dtype(*dtype);
        if (layout.has_value()) opts_override = opts_override.layout(*layout);
        if (device.has_value()) opts_override = opts_override.device(*device);
        if (pin_memory.has_value()) opts_override = opts_override.pinned_memory(*pin_memory);

        c10::TensorOptions options = self.options().merge_in(opts_override);

        // 2) layout 不能改变（与 NPU 实现一致）
        if (layout.has_value()) {
            TORCH_CHECK(
                self.layout() == layout.value(),
                "to(options) doesn't support converting to a different layout, "
                "but got self.layout being ",
                self.layout(),
                " and options.layout set as ",
                layout.value());
        }

        // 3) 规范化 device index（若提供）
        if (device.has_value()) {
            options = options.device(ensure_has_index(*device));
        }

        // 4) dtype double on OCL fallback to float (consistent with to(dtype) behavior)
        if (options.dtype() == at::ScalarType::Double && options.device().type() == OpenCLDeviceType) {
            if (!CLContextManager::fp64(options.device().index())) {
                TORCH_WARN("Device ocl:" + std::to_string(options.device().index()) + " does not support cl_khr_fp64, falling back to float");
                options = options.dtype(at::ScalarType::Float);
            }
        }

        // 5) requires_grad must not be set in options
        TORCH_CHECK(
            options.requires_grad_opt() == c10::nullopt,
            "to(options) expects unset requires_grad flag, but got "
            "options.requires_grad set as ",
            options.requires_grad());

        // 6) memory_format handling: determine effective memory_format
        c10::MemoryFormat mf = memory_format.has_value()
            ? *memory_format
            : (self.device().type() == OpenCLDeviceType ? c10::MemoryFormat::Contiguous : c10::MemoryFormat::Preserve);

        if (memory_format.has_value()) {
            TORCH_CHECK(
                mf == c10::MemoryFormat::Preserve || mf == c10::MemoryFormat::Contiguous,
                "Only contiguous_format or preserve_format is supported");
            // options already has memory_format? we'll set below when creating tensors
        } else {
            // options.memory_format will be applied later
        }

        // 7) pin_out behavior (non_blocking -> CPU pinned) similar to to_impl_dlprim
        bool pin_out = non_blocking && (self.device().type() == OpenCLDeviceType) && options.device().is_cpu() &&
                       (options.layout() == c10::kStrided);

        // 8) If Preserve requested and tensor is non-overlapping and dense, use empty_strided to preserve strides
        if (mf == c10::MemoryFormat::Preserve) {
            if (self.is_non_overlapping_and_dense()) {
                auto r = at::empty_strided(
                    self.sizes(), self.strides(),
                    options.memory_format(c10::nullopt).pinned_memory(pin_out));
                r.copy_(self, non_blocking);
                return r;
            } else {
                // fallback to suggested memory format
                mf = self.suggest_memory_format();
            }
        }

        // 9) default: allocate empty with chosen memory_format and copy
        auto r = at::empty(
            self.sizes(),
            options.memory_format(mf).pinned_memory(pin_out),
            c10::nullopt);
        r.copy_(self, non_blocking);
        return r;
    }


    Tensor &fill_(Tensor &self, const c10::Scalar &value)
    {
        GUARD;
        dlprim::Tensor t(todp(self));
        auto q = getExecutionContext(self);
        dlprim::core::fill_tensor(t,value.to<double>(),q);
        sync_if_needed(self.device());
        return self;
    }
    
    Tensor &zero_(Tensor &self)
    {
        GUARD;
        if(self.numel() == 0)
            return self;
        Tensor self_c = self.contiguous();
        dlprim::Tensor t(todp(self));
        dlprim::core::fill_tensor(t,0.0,getExecutionContext(self));
        if(!self.is_contiguous())
            self.copy_(self_c);
        return self;
    }

    Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
    {
        GUARD;
        Tensor result = at::alias(self);
        result.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        if(storage_offset)
            result.getIntrusivePtr()->set_storage_offset(*storage_offset);
        return result;

    }

    // {"schema": "aten::_local_scalar_dense(Tensor self) -> Scalar", "dispatch": "True", "default": "False"}
    Scalar _local_scalar_dense(const Tensor & self)
    {
        GUARD;
        TORCH_CHECK(self.numel()==1);
        dlprim::Tensor x=todp(self);
        union {
            float f;
            double d;
            int8_t i8;
            uint8_t u8;
            int16_t i16;
            uint16_t u16;
            int32_t i32;
            uint32_t u32;
            int64_t i64;
            uint64_t u64;
            char data[16];
        } data;
        x.to_host(getExecutionContext(self),data.data);
        switch(x.dtype()) {
        case dlprim::float_data:    return data.f;
        case dlprim::double_data:   return data.d;
        case dlprim::int8_data:     return data.i8;
        case dlprim::uint8_data:    return data.u8;
        case dlprim::int16_data:    return data.i16;
        case dlprim::uint16_data:   return data.u16;
        case dlprim::int32_data:    return (int64_t)data.i32;
        case dlprim::uint32_data:   return (int64_t)data.u32;
        case dlprim::int64_data:    return (int64_t)data.i64;
        case dlprim::uint64_data:   return (int64_t)data.u64;
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented data type");
        }
    }

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
    
    // {"schema": "aten::masked_select(Tensor self, Tensor mask) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor masked_select(const Tensor & self, const Tensor & mask)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        Tensor mask_c = mask.contiguous();
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
        Tensor res=new_tensor_as(dlprim::Shape(N),self);
        if(N > 0) {
            dlprim::Tensor y=todp(res);
            y.to_device(getExecutionContext(self),x.host_data());
        }
        sync_if_needed(self.device());
        return res;
    }
   
    // {"schema": "aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride)
    {
        c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr(); 
        impl->set_storage_keep_dtype(source);
        impl->set_sizes_and_strides(size, stride, storage_offset);
        return self;
    }

    // {"schema": "aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & set_(Tensor & self, Storage source)
    {
        c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr(); 
        auto size = source.nbytes();
        impl->set_storage_keep_dtype(source);
        int elem_size = torch::elementSize(torch::typeMetaToScalarType(self.dtype()));
        std::vector<int64_t> vsizes = { int64_t(size / elem_size) };
        c10::ArrayRef<int64_t> sizes(vsizes.data(),vsizes.size());
        impl->set_sizes_contiguous(sizes);

        return self;
    }

    Tensor & set_(Tensor & self, const Tensor & source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride)
    {

        at::TensorImpl* self_impl  = self.unsafeGetTensorImpl();
        at::TensorImpl* src_impl   = source.unsafeGetTensorImpl();

        if (self_impl == src_impl) {
            return self;
        }

        // ---- 2. 基本合法性检查（仿 NPU 做法）----
        // 如果你有 checkSetStorage，可以启用：
        // at::native::checkSetStorage(self, source.storage(), storage_offset, size, stride);

        // ---- 3. 替换 storage（保持 dtype）----
        // 与 NPU 的 set_storage_nd_npu 行为完全一致
        self_impl->set_storage_keep_dtype(source.storage());

        // ---- 4. 设置 storage offset ----
        self_impl->set_storage_offset(storage_offset);

        // ---- 5. 设置 sizes / strides ----
        // 若 stride 为空（例如 { }），你可以决定是否自动 contiguous
        if (stride.size() > 0) {
            // 按指定 stride
            self_impl->set_sizes_and_strides(size, stride);
        } else {
            // 默认 contiguous（与 NPU 等效行为）
            self_impl->set_sizes_contiguous(size);
        }

        // ---- 6. backend-specific storage descriptor 同步 ----
        // 如果你的 OpenCL 后端有类似 NPU 的 StorageDescHelper，请在此处调用：
        // StorageDescHelper::CopyDesc(self, source);

        // 如果还没有 descriptor helper，可以忽略这步，不影响基本功能。

        return self;
    }

   // {"schema": "aten::set_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & set_(Tensor & self) {
        GUARD;

        // 保存 dtype（与 NPU 实现一致，后面断言不被改变）
        caffe2::TypeMeta dtype = self.dtype();

        // 使用 CLContextManager 分配 0 字节的 DataPtr（作为 backend storage）
        // CLContextManager::allocate(device, nbytes) 返回 at::DataPtr
        at::DataPtr data_ptr = CLContextManager::allocate(self.device(), /*n=*/0);

        // 创建 StorageImpl：传入 nullptr 作为 allocator（你的 CLContextManager 无 get_allocator）
        // storage size 表示以字节为单位，这里用 SymInt(0)
        c10::intrusive_ptr<c10::StorageImpl> ocl_storage_impl =
        c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            c10::SymInt(0),
            std::move(data_ptr),
            /*allocator=*/ static_cast<c10::Allocator*>(nullptr),
            /*resizable=*/ true);

        // 构造 c10::Storage
        c10::Storage storage(ocl_storage_impl);

        // 将 self 指向这个 storage（保持 dtype）
        c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr();
        impl->set_storage_keep_dtype(storage);

        // storage offset = 0
        impl->set_storage_offset(0);

        // 设置 sizes = {0}；stride 为空 -> 设为 contiguous
        std::vector<int64_t> sizes = {0};
        impl->set_sizes_contiguous(c10::ArrayRef<int64_t>(sizes.data(), sizes.size()));

        // 如果你有 backend-specific descriptor helper，可在此初始化：
        // e.g. StorageDescHelper::SetDesc(self);

        // 确认 dtype 未被修改
        TORCH_INTERNAL_ASSERT(dtype == self.dtype(), "set_(self) changed dtype unexpectedly");

        return self;
    }


    // {"schema": "aten::resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}    
    const Tensor & resize_(const Tensor & self, at::IntArrayRef size, ::std::optional<at::MemoryFormat> memory_format)
    {
        if(memory_format) {
            TORCH_CHECK(*memory_format == MemoryFormat::Contiguous, "resize_ only supports contiguous memory format");
        }
        c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr();
        c10::Storage const &storage = impl->storage();
        int64_t storage_size = storage.nbytes();
        at::DataPtr &data = storage.mutable_data_ptr();

        int64_t new_size = 1;
        std::vector<int64_t> vsizes;
        vsizes.reserve(size.size());
        for (int64_t dim : size) {
            new_size *= dim;
            vsizes.push_back(dim);
        }
        c10::ArrayRef<int64_t> sizes(vsizes.data(),vsizes.size());

        dlprim::DataType dt = todp(self.dtype());
        new_size*=dlprim::size_of_data_type(dt);
        
        if(new_size >= storage_size && new_size > 0) {
            at::DataPtr new_mem = CLContextManager::allocate(self.device(),new_size);
            if(storage_size > 0) {
                cl::Buffer dst((cl_mem)new_mem.get(),true);
                cl::Buffer src((cl_mem)data.get(),true);
                auto q = getExecutionContext(self);
                q.queue().enqueueCopyBuffer(src,dst,0,0,storage_size,q.events(),q.event("copy_buffer"));
            } 
            data = std::move(new_mem);
            storage.set_nbytes(new_size);
            sync_if_needed(self.device());
        }
        impl->set_sizes_contiguous(sizes);
        return self;
    }

    

    // to_impl_dlprim: 统一实现 copy/create 的逻辑（类似 to_impl_npu）
    static inline at::Tensor to_impl_dlprim(
        const at::Tensor& self,
        const c10::TensorOptions& options,
        bool non_blocking,
        bool copy) 
    {
        GUARD;

        // memory_format default: Preserve on CPU-like backends? 保持与上面样例一致的策略
        auto memory_format = options.memory_format_opt().value_or(c10::MemoryFormat::Contiguous);

        if (self.dtype() == options.dtype() &&
            self.layout() == options.layout() &&
            self.device() == options.device() && !copy &&
            (memory_format == c10::MemoryFormat::Preserve || self.suggest_memory_format() == memory_format)) {
            return self;
        }

        bool pin_out = non_blocking && (self.device().type() == OpenCLDeviceType) && options.device().is_cpu() &&
                       (options.layout() == c10::kStrided);

        // Preserve behavior: 如果用户要求 Preserve，并且 tensor 是 non-overlapping AND dense，
        // 则复制 strides；否则退化为建议的 memory_format
        if (memory_format == c10::MemoryFormat::Preserve) {
            if (self.is_non_overlapping_and_dense()) {
                auto r = at::empty_strided(
                    self.sizes(), self.strides(),
                    options.memory_format(c10::nullopt).pinned_memory(pin_out));
                r.copy_(self, non_blocking);
                return r;
            } else {
                memory_format = self.suggest_memory_format();
            }
        }

        // 其它情况：创建空 tensor 再 copy
        auto r = at::empty(
            self.sizes(), options.memory_format(memory_format).pinned_memory(pin_out), c10::nullopt);
        r.copy_(self, non_blocking);
        return r;
    }

    /* {"schema": "to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)" */
    at::Tensor to(
        const at::Tensor& self,
        c10::Device device,
        at::ScalarType dtype,
        bool non_blocking,
        bool copy,
        c10::optional<c10::MemoryFormat> optional_memory_format)
    {
        GUARD;
        device = ensure_has_index(device);

        c10::TensorOptions opts = self.options().device(device).dtype(dtype);
        if (optional_memory_format.has_value()) {
            opts = opts.memory_format(optional_memory_format.value());
        }
        return to_impl_dlprim(self, opts, non_blocking, copy);
    }

    /* "schema": "to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)" */
    at::Tensor to(
        const at::Tensor& self,
        at::ScalarType dtype,
        bool non_blocking,
        bool copy,
        c10::optional<c10::MemoryFormat> optional_memory_format)
    {
        GUARD;

        if (self.dtype() == dtype && !copy) {
            return self;
        }


        if (dtype == at::ScalarType::Double) {
            bool fp64_supported = true;
            if (self.device().type() == OpenCLDeviceType) {
                fp64_supported = CLContextManager::fp64(self.device().index());
            }
            if (!fp64_supported) {
                TORCH_WARN("Device ocl:" + std::to_string(self.device().index()) + " does not support double, casting to float.");
                dtype = at::ScalarType::Float;
            }
        }


        TensorOptions opts = self.options().dtype(dtype);
        if (optional_memory_format.has_value()) {
            opts = opts.memory_format(optional_memory_format.value());
        }

        Tensor tmp = at::empty(self.sizes(), opts);
        tmp.copy_(self, non_blocking);
        return tmp;
    }

    /* {"schema": "Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"}*/
    at::Tensor to(
        const at::Tensor& self,
        const at::Tensor& other,
        bool non_blocking,
        bool copy,
        c10::optional<c10::MemoryFormat> optional_memory_format)
    {
        GUARD;
        c10::TensorOptions opts = other.options();
        if (optional_memory_format.has_value()) {
            opts = opts.memory_format(optional_memory_format.value());
        }
        return to_impl_dlprim(self, opts, non_blocking, copy);
    }

    bool _has_compatible_shallow_copy_type(const at::Tensor & self, const at::Tensor & from) {
        c10::DispatchKeySet self_keyset = self.key_set();
        c10::DispatchKeySet from_keyset = from.key_set();
        auto is_dense = [](c10::DispatchKeySet ks) {
            return ks.has(c10::DispatchKey::CPU) || ks.has(c10::DispatchKey::PrivateUse1);
        };
        return (self_keyset == from_keyset) || (is_dense(self_keyset) && is_dense(from_keyset));
    }

  }  /* namespace op_plugin */
}  /* namespace at_torch */
