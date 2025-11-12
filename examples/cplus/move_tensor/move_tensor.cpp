#include <iostream>
#include <dlfcn.h>
#include <torch/torch.h>

int main() {
    const char* lib_path = "/home/kylin/gjl/project/pytorch_dlprim/build/debug/pytorch_ocl/pt_ocl.so";
    
    // 打开动态库
    void* handle = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "无法加载动态库: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "动态库加载成功: " << lib_path << std::endl;
    dlclose(handle);

    float a[2] = {0.2f, 0.3f};
    torch::Tensor tensor = torch::from_blob(a, {2});
    std::cout << "tensor: " << tensor << std::endl;

    torch::register_privateuse1_backend("ocl");
    torch::Device device("ocl:0");
    torch::Tensor ocl_tensor = tensor.to(device); //THIS LINE CAUSES SEGFAULT
    std::cout << "ocl tensor: " << ocl_tensor << std::endl;

    return 0;
}
