#include "OpInterface.h"

namespace at_torch
{
namespace op_plugin
{

const at::Tensor& resize_(
    const at::Tensor& self, at::IntArrayRef size, ::std::optional<at::MemoryFormat> memory_format)
{
    if (memory_format)
    {
        TORCH_CHECK(
            *memory_format == MemoryFormat::Contiguous,
            "resize_ only supports contiguous memory format");
    }
    c10::intrusive_ptr<c10::TensorImpl> impl = self.getIntrusivePtr();
    c10::Storage const& storage = impl->storage();
    int64_t storage_size = storage.nbytes();
    at::DataPtr& data = storage.mutable_data_ptr();

    int64_t new_size = 1;
    std::vector<int64_t> vsizes;
    vsizes.reserve(size.size());
    for (int64_t dim : size)
    {
        new_size *= dim;
        vsizes.push_back(dim);
    }
    c10::ArrayRef<int64_t> sizes(vsizes.data(), vsizes.size());

    dlprim::DataType dt = todp(self.dtype());
    new_size *= dlprim::size_of_data_type(dt);

    if (new_size >= storage_size && new_size > 0)
    {
        at::DataPtr new_mem = CLContextManager::allocate(self.device(), new_size);
        if (storage_size > 0)
        {
            cl::Buffer dst((cl_mem)new_mem.get(), true);
            cl::Buffer src((cl_mem)data.get(), true);
            auto q = getExecutionContext(self);
            q.queue().enqueueCopyBuffer(
                src, dst, 0, 0, storage_size, q.events(), q.event("copy_buffer"));
        }
        data = std::move(new_mem);
        storage.set_nbytes(new_size);
        sync_if_needed(self.device());
    }
    impl->set_sizes_contiguous(sizes);
    return self;
}

} /* namespace op_plugin */
} /* namespace at_torch */
