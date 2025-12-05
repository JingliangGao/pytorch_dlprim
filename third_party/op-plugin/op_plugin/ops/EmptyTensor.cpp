#include "OpInterface.h"

namespace at_torch {
namespace op_plugin {


    at::Tensor empty(IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format)
    {

        TORCH_CHECK(!layout || *layout == Layout::Strided,"torch_kpu supports only strided layout")
        // FIX ME Later -how to handle non Contiguous format {
        //TORCH_CHECK(!memory_format || *memory_format == MemoryFormat::Contiguous,"Contigonous format expected");
        // }
        c10::Device dev = device ? *device : c10::Device(c10::DeviceType::PrivateUse1,0);
        c10::ScalarType st = dtype ? *dtype : c10::kFloat;
        if(st == c10::kDouble && !CLContextManager::fp64(dev.index())) {
            st = c10::kFloat;
            TORCH_WARN("This device ocl:" + std::to_string(dev.index()) + " does not support cl_khr_fp64, falling back to float");
        }
        return at_torch::new_ocl_tensor(size,dev,st);
    }


    at::Tensor empty_strided(
        at::IntArrayRef size,
        at::IntArrayRef stride,
        ::std::optional<at::ScalarType> dtype,
        ::std::optional<at::Layout> layout,
        ::std::optional<at::Device> device,
        ::std::optional<bool> pin_memory)
    {

        at::Tensor r = empty(size,dtype,layout,device,pin_memory,c10::nullopt);
        at::Tensor data = at::alias(r);
        data.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        return data;
    }


    }  /* namespace op_plugin */
}  /* namespace at_torch */
