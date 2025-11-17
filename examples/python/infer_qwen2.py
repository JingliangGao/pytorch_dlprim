import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import pytorch_ocl
import time
import argparse

# -----------------------------
# Tokenizer
# -----------------------------
# 加载官方 tokenizer.json
tokenizer = Tokenizer.from_file("<tokenizer-json-file>")
VOCAB_SIZE = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)

# -----------------------------
# 配置参数 — 根据模型规格修改
# -----------------------------
HIDDEN_SIZE = 1024          # 隐藏层维度
NUM_LAYERS = 24             # Transformer 层数
NUM_HEADS = 16              # 注意力头数
MAX_SEQ_LEN = 32768         # 最大上下文长度
DEVICE = "ocl"


# -----------------------------
# RMSNorm 层
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)

# -----------------------------
# FeedForward 层（SwiGLU 简化）
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_multiplier=4):
        super().__init__()
        inner_dim = hidden_size * ff_multiplier
        self.fc_in = nn.Linear(hidden_size, inner_dim, bias=True)
        self.fc_gate = nn.Linear(hidden_size, inner_dim, bias=True)
        self.fc_out = nn.Linear(inner_dim, hidden_size, bias=True)

    def forward(self, x):
        return self.fc_out(F.silu(self.fc_in(x)) * self.fc_gate(x))

# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.ln1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.ln2 = RMSNorm(hidden_size)
        self.ff = FeedForward(hidden_size)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

# -----------------------------
# Qwen 模型
# -----------------------------
class QwenForCausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.ln_f = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        B, T = input_ids.size()
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# 生成函数（贪心）
# -----------------------------
def generate(model, prompt, max_new_tokens=50, temperature=1.0):
    model.eval()
    input_ids = torch.tensor([encode(prompt)], device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[:, -1, :] / temperature
        next_id = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    output_ids = input_ids[0].tolist()
    return decode(output_ids)

def parse_argument():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--profile', type=str, default=None,
                        help='Path to save profiling data')
    args = parser.parse_args()
    return args
    
def main():
    # -----------------------------
    # 加载模型权重
    # -----------------------------
    model = QwenForCausalLM(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN).to(DEVICE)
    state_dict = torch.load("<weight-pt-file>", map_location=DEVICE)
    # 打印前 10 个 key，方便调试
    print("Loaded state_dict keys sample:", list(state_dict.keys())[:10])
    model.load_state_dict(state_dict, strict=False)  # strict=False 避免部分键名不匹配
    model.eval()

    prompt = "你好啊"
    ts = time.time()
    output = generate(model, prompt, max_new_tokens=100)
    te = time.time()
    print("生成结果:\n", output)
    print(f"All finished, total cost time : {te - ts} s.")

# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    args = parse_argument()
    if args.profile:
        torch.ocl.enable_profiling('ocl')
        with torch.ocl.profile('ocl', args.profile):
            main()
    else:
        main()
    
