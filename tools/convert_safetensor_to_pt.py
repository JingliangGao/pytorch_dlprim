from transformers import AutoModelForCausalLM
from pathlib import Path
import torch
import argparse


def parse_argument():
    parser = argparse.ArgumentParser(description="Convert Safetensor to PT")
    parser.add_argument(
        "--model", type=str, help="Model directory containing safetensor files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the converted PT file",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="entire",
        choices=["state_dict", "entire"],
        help="Type of the model file",
    )
    args = parser.parse_args()
    return args


def convert_safetensor_to_pt(args):
    # 1. 自动从多个 safetensors 加载模型（Transformers 会自动合并）
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="float32", device_map="cpu"
    )

    content = None
    model_name = Path(args.model).name
    if args.type == "state_dict":
        model = model.to("cpu")  # 确保模型在 CPU 上
        content = model.state_dict()
        model_name = model_name + "_state_dict.pt"
    else:
        content = model
        model_name = model_name + ".pt"

    # 3. save model
    current_file = Path(__file__).resolve()
    data_dir = f"{current_file.parent.parent}/examples/data/"
    if args.output is not None:
        data_dir = args.output
    torch.save(content, f"{data_dir}/{model_name}")

    print(f"[INFO] Saved merged '{model_name}' in {data_dir} . ")


if __name__ == "__main__":
    args = parse_argument()
    convert_safetensor_to_pt(args)
