#include <dlfcn.h>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


const char* lib_path = "<path-to-libpt_ocl.so>";
const std::string scripted_model = "<path-to-scripted_model.pt>";

void infer_net() {
    /* load dynamic library */
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
        exit(1);
    }
    std::cout << "Dynamic library loaded successfully: " << lib_path << std::endl;

    /* register ocl backend */
    torch::register_privateuse1_backend("ocl");
    torch::Device device("cpu");    /* option: "cpu", "ocl:0" */

    /* create dummy input */
    torch::Tensor input = torch::randn({1, 1, 28, 28}).to(device);

    /* load scripted module */
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

int main() {
    infer_net();
    return 0;
} 
