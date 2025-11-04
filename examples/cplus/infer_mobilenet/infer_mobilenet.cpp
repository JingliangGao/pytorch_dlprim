#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>


int main() {
    torch::Device device(torch::kCPU);

    try {
        device = torch::Device(torch::kPrivateUse1);
        std::cout << "✅ 使用 PrivateUse1 设备推理\n";
    } catch (const c10::Error& e) {
        std::cout << "⚠️ 未检测到 PrivateUse1 后端，使用 CPU\n";
        device = torch::Device(torch::kCPU);
    }

    // ✅ 加载 TorchScript 模型并移动到 device
    std::string pt_file = std::string(DATA_DIR) + "mobilenetv2_ts.pt";
    std::cout << "pt_file : " << pt_file << std::endl;
    torch::jit::script::Module model = torch::jit::load(pt_file);
    model.to(device);
    model.eval();

    // ✅ 读 MNIST 图片
    cv::Mat img = cv::imread("mnist0.png", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(224, 224));
    cv::Mat img3;
    cv::cvtColor(img, img3, cv::COLOR_GRAY2BGR);

    // ✅ 创建 tensor
    torch::Tensor input = torch::from_blob(
        img3.data, {1, 224, 224, 3}, torch::kUInt8
    ).permute({0, 3, 1, 2}).to(torch::kFloat).div(255.0);

    // ✅ normalize
    input = (input - torch::tensor({0.485, 0.456, 0.406}).view({1,3,1,1}))
                  / torch::tensor({0.229, 0.224, 0.225}).view({1,3,1,1});

    // ✅ 放入 PrivateUse1
    input = input.to(device);

    // ✅ 推理
    torch::NoGradGuard no_grad;
    torch::Tensor out = model.forward({input}).toTensor();

    // ✅ 拿回 CPU 输出
    out = out.to(torch::kCPU);
    int pred = out.argmax(1).item<int>();

    std::cout << "预测结果 = " << pred << std::endl;
    return 0;
}
