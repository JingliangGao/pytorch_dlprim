#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {

    // {"schema": "aten::set_.source_Storage_storage_offset(at::Tensor(a!) self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> at::Tensor(a!)", "dispatch": "True", "default": "False"}
    at::Tensor & set_(at::Tensor & self, Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride)
    {
        c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr(); 
        impl->set_storage_keep_dtype(source);
        impl->set_sizes_and_strides(size, stride, storage_offset);
        return self;
    }

    // {"schema": "aten::set_.source_Storage(at::Tensor(a!) self, Storage source) -> at::Tensor(a!)", "dispatch": "True", "default": "False"}
    at::Tensor & set_(at::Tensor & self, Storage source)
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

    at::Tensor & set_(at::Tensor & self, const at::Tensor & source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride)
    {

        at::TensorImpl* self_impl  = self.unsafeGetTensorImpl();
        at::TensorImpl* src_impl   = source.unsafeGetTensorImpl();

        if (self_impl == src_impl) {
            return self;
        }

            // ---- 2. Basic validity checks (following NPU behavior) ----
            // If you have checkSetStorage, you can enable it:
            // at::native::checkSetStorage(self, source.storage(), storage_offset, size, stride);

            // ---- 3. Replace storage (keep dtype) ----
            // This behavior matches NPU's set_storage_nd_npu exactly
        self_impl->set_storage_keep_dtype(source.storage());

            // ---- 4. Set storage offset ----
        self_impl->set_storage_offset(storage_offset);

            // ---- 5. Set sizes / strides ----
            // If stride is empty (e.g. { }), you may choose to make the tensor contiguous automatically
        if (stride.size() > 0) {
                // Use the specified stride
            self_impl->set_sizes_and_strides(size, stride);
        } else {
                // Default to contiguous (equivalent to NPU behavior)
            self_impl->set_sizes_contiguous(size);
        }

            // ---- 6. Sync backend-specific storage descriptor ----
            // If your OpenCL backend has a StorageDescHelper similar to NPU's, call it here:
            // StorageDescHelper::CopyDesc(self, source);

            // If no descriptor helper exists yet, this step can be skipped without affecting basic functionality.

        return self;
    }

   // {"schema": "aten::set_(at::Tensor(a!) self) -> at::Tensor(a!)", "dispatch": "True", "default": "False"}
    at::Tensor & set_(at::Tensor & self) {
        GUARD;

        // Preserve dtype (same as NPU implementation; later assert it is unchanged)
        caffe2::TypeMeta dtype = self.dtype();

        // Use CLContextManager to allocate a 0-byte DataPtr (used as backend storage)
        // CLContextManager::allocate(device, nbytes) returns an at::DataPtr
        at::DataPtr data_ptr = CLContextManager::allocate(self.device(), /*n=*/0);

        // Create a StorageImpl: pass nullptr for allocator (your CLContextManager does not provide get_allocator)
        // Storage size is in bytes; use SymInt(0) here
        c10::intrusive_ptr<c10::StorageImpl> ocl_storage_impl =
        c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            c10::SymInt(0),
            std::move(data_ptr),
            /*allocator=*/ static_cast<c10::Allocator*>(nullptr),
            /*resizable=*/ true);

        // Construct c10::Storage
        c10::Storage storage(ocl_storage_impl);

        // Point self to this storage (keep dtype)
        c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr();
        impl->set_storage_keep_dtype(storage);

        // storage offset = 0
        impl->set_storage_offset(0);

    // Set sizes = {0}; stride is empty -> set as contiguous
        std::vector<int64_t> sizes = {0};
        impl->set_sizes_contiguous(c10::ArrayRef<int64_t>(sizes.data(), sizes.size()));

    // If you have a backend-specific descriptor helper, initialize it here:
    // e.g. StorageDescHelper::SetDesc(self);

    // Verify dtype was not modified
        TORCH_INTERNAL_ASSERT(dtype == self.dtype(), "set_(self) changed dtype unexpectedly");

        return self;
    }

  }  /* namespace op_plugin */
}  /* namespace at_torch */