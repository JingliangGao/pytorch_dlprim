#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>


void infer_net_torchscript(const torch::Device& device)
{

    /* create dummy input */
    torch::Tensor input = torch::randn({1, 3, 224, 224}).to(device);
    // std::cout << "Input : " << input << std::endl;

    /* load scripted module */
    std::string scripted_model = "<path-to-mobilenetv2_trace.pt>";
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
    torch::Device device(torch::kPrivateUse1, 0); /* backend : torch::kPrivateUse1, torch::kCPU  */

    auto start = std::chrono::steady_clock::now();
    int NUM = 50;
    for (int i = 0; i < NUM; i++)
    {
        infer_net_torchscript(device);
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "cost time : " << duration.count() << " ms\n";
    return 0;
}
