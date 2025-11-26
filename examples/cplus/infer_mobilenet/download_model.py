
import torch
import torchvision
from pathlib import Path

def export_mobilenetv2_torchscript( use_script=False):
    # 1. download mobilenetv2 model
    model = torchvision.models.mobilenet_v2(weights="DEFAULT")
    model.eval()

    # 2. prepare example input (needed for trace)
    example = torch.randn(1, 3, 224, 224)

    # 3. choose trace or script
    if use_script:
        print("Using torch.jit.script...")
        ts_model = torch.jit.script(model)
    else:
        print("Using torch.jit.trace...")
        ts_model = torch.jit.trace(model, example)

    # 4. save TorchScript
    file_name = "mobilenetv2_script.pt" if use_script else "mobilenetv2_trace.pt"
    current_file = Path(__file__).resolve()   
    examples_dir = current_file.parent.parent.parent 
    ts_model.save(examples_dir / "data" / file_name)
    print(f"TorchScript model saved to {file_name}")

if __name__ == "__main__":
    export_mobilenetv2_torchscript(use_script=False)

