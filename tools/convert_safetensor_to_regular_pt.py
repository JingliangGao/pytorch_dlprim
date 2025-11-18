from transformers import AutoModelForCausalLM
from pathlib import Path
import torch
import argparse


def parse_argument():
    parser = argparse.ArgumentParser(description='Convert Safetensor to PT')
    parser.add_argument('--model', type=str, help='Model directory containing safetensor files')
    parser.add_argument('--output', type=str, default=None, help='Output directory for the converted PT file')
    args = parser.parse_args()
    return args

def convert_safetensor_to_pt(args):
    # 1. 自动从多个 safetensors 加载模型（Transformers 会自动合并）
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="float32", device_map="cpu")

    # 2. 提取合并后的 state_dict
    state_dict = model.state_dict()

    # 3. 保存为单一 pt
    model_name = Path(args.model).name
    current_file = Path(__file__).resolve()   
    data_dir = f"{current_file.parent.parent}/examples/data/"
    if args.output is not None:
        data_dir = args.output
    torch.save(state_dict, f"{data_dir}/{model_name}_state_dict.pt")

    print(f"[INFO] Saved merged '{model_name}_state_dict.pt' in {data_dir} . ")

if __name__ == "__main__":
    args = parse_argument()
    convert_safetensor_to_pt(args)
