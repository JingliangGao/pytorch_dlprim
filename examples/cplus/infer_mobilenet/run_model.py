import torch
import torch_kpu
import time

def infer_net_torchscript():
    # 选择设备
    device = torch.device("kpu")

    # 创建 dummy 输入
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # 加载 TorchScript 模型
    scripted_model_path = "/home/kylin/gjl/project/pytorch_dlprim/examples/data/mobilenetv2_trace.pt"
    print(f"Loading scripted module: {scripted_model_path}")

    model = torch.jit.load(scripted_model_path)
    model.to(device)
    model.eval()

    # 推理
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.exp(out)           # 与 C++ 等价
        _, top = probs.max(1)            # 取最大值的索引

    print("Predicted class (scripted):", top.item())


if __name__ == "__main__":
    ts = time.time()
    NUM = 50
    for i in range(0, NUM):
        infer_net_torchscript()
    te = time.time()
    print("Cost time : ", te - ts)
