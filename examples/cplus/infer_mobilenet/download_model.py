import torch
import torchvision.models as models
from pathlib import Path

# 下载预训练 MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

current_file = Path(__file__).resolve()   
examples_dir = current_file.parent.parent.parent 
traced.save(f"{examples_dir}/data/mobilenetv2_ts.pt")

print("✅ saved mobilenetv2_ts.pt")
