#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


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


    at::Tensor _to_copy(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, ::std::optional<at::MemoryFormat> memory_format){
        GUARD;

        // 1) create override options then merge
        c10::TensorOptions opts_override;
        if (dtype.has_value()) opts_override = opts_override.dtype(*dtype);
        if (layout.has_value()) opts_override = opts_override.layout(*layout);
        if (device.has_value()) opts_override = opts_override.device(*device);
        if (pin_memory.has_value()) opts_override = opts_override.pinned_memory(*pin_memory);

        c10::TensorOptions options = self.options().merge_in(opts_override);

        // 2) can not change layout
        if (layout.has_value()) {
            TORCH_CHECK(
                self.layout() == layout.value(),
                "to(options) doesn't support converting to a different layout, "
                "but got self.layout being ",
                self.layout(),
                " and options.layout set as ",
                layout.value());
        }

        // 3) check device index
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

        at::Tensor tmp = at::empty(self.sizes(), opts);
        tmp.copy_(self, non_blocking);
        return tmp;
    }


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

    }  /* namespace op_plugin */
}  /* namespace at_torch */
