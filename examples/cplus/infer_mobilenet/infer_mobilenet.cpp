#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

torch::Device device(torch::kCPU, 0); /* backend : torch::kPrivateUse1, torch::kCPU  */

void load_device()
{
    const char* lib_path = "/home/kylin/gjl/project/pytorch_dlprim/build/debug/"
                           "torch_kpu/libtorch_kpu.so";
    /* load dynamic library */
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle)
    {
        std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
        exit(1);
    }
    std::cout << "Dynamic library loaded successfully: " << lib_path << std::endl;
}

void infer_net_torchscript()
{
    /* create dummy input */
    torch::Tensor input = torch::randn({1, 3, 224, 224}).to(device);
    // std::cout << "Input : " << input << std::endl;

    /* load scripted module */
    std::string scripted_model = "/home/kylin/gjl/project/pytorch_dlprim/"
                                 "examples/data/mobilenetv2_trace.pt";
    std::cout << "Loading scripted module: " << scripted_model << std::endl;
    auto module = torch::jit::load(scripted_model);
    module.to(device);
    module.eval();

    /* inference */
    auto out = module.forward({input}).toTensor();
    auto probs = torch::exp(out);
    auto top = std::get<1>(probs.max(1));
    std::cout << "Predicted class (scripted): " << top.item<int>() << std::endl;
}

int main()
{
    load_device();
    auto start = std::chrono::steady_clock::now();
    int NUM = 50;
    for (int i = 0; i < NUM; i++)
    {
        infer_net_torchscript();
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "耗时: " << duration.count() << " ms\n";
    return 0;
}
