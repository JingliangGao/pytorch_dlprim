#include <iostream>
#include <dlfcn.h>
#include <torch/torch.h>
#include <torch/script.h> 

const char* lib_path = "<path-to-libpt_ocl.so>";

/* create a Mnist net */
struct Net : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)));
        fc1 = register_module("fc1", torch::nn::Linear(9216, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    /* forward */
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1}); // flatten
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }
}; 


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
    torch::Device device("ocl:0");

    /* load tensor */
    float a[2] = {0.2f, 0.3f};
    torch::Tensor tensor = torch::from_blob(a, {2});
    torch::Tensor ocl_tensor = tensor.to(device);
    std::cout << "ocl tensor: " << ocl_tensor << std::endl;

    /* load net */
    Net net;
    net.to(device);
}

int main() {
    infer_net();
    return 0;
} 