from transformers import AutoModelForCausalLM
from pathlib import Path
import torch

model_dir = "<Your-safetensors-model-directory>"

# 1. 自动从多个 safetensors 加载模型（Transformers 会自动合并）
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="float32", device_map="cpu")

# 2. 提取合并后的 state_dict
state_dict = model.state_dict()

# 3. 保存为单一 pt
current_file = Path(__file__).resolve()   
data_dir = f"{current_file.parent.parent}/examples/data/"
model_name = Path(model_dir).name
torch.save(state_dict, f"{data_dir}/{model_name}_state_dict.pt")

print(f"[INFO] Saved merged '{model_name}_state_dict.pt' in {data_dir} . ")