// infer_qwen2.cpp

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

// -----------------------------
// Config
// -----------------------------
const int64_t HIDDEN_SIZE = 1024;
const int64_t NUM_LAYERS = 24;
const int64_t NUM_HEADS = 16;
const int64_t MAX_SEQ_LEN = 32768;
const int64_t VOCAB_SIZE = 150000;
const torch::Device DEVICE = torch::kCPU;

// -----------------------------
// Placeholder Tokenizer
// -----------------------------
std::vector<int64_t> encode(const std::string& text)
{
    std::vector<int64_t> ids;
    for (unsigned char c : text)
        ids.push_back(c % VOCAB_SIZE);
    if (ids.empty())
        ids.push_back(0);
    return ids;
}

std::string decode(const std::vector<int64_t>& ids)
{
    std::string s;
    for (auto id : ids)
        s.push_back((char)(id % 256));
    return s;
}

// -----------------------------
// RMSNorm
// -----------------------------
struct RMSNormImpl : torch::nn::Module
{
    torch::Tensor weight;
    double eps;
    int64_t hidden_size;

    RMSNormImpl(int64_t hidden_size_, double eps_ = 1e-8) : eps(eps_), hidden_size(hidden_size_)
    {
        weight = register_parameter("weight", torch::ones({hidden_size_}));
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        auto rms = x.pow(2).mean(-1, true).add(eps).sqrt();
        return x / rms * weight;
    }
};

// -----------------------------
// MultiHeadAttention
// -----------------------------
struct MultiHeadAttentionImpl : torch::nn::Module
{
    int64_t num_heads;
    int64_t head_dim;
    int64_t hidden_size;
    double scale;

    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};

    MultiHeadAttentionImpl(int64_t hidden_size_, int64_t num_heads_)
        : num_heads(num_heads_), hidden_size(hidden_size_)
    {
        head_dim = hidden_size_ / num_heads_;
        scale = 1.0 / std::sqrt((double)head_dim);

        q_proj = register_module("q_proj", torch::nn::Linear(hidden_size_, hidden_size_));
        k_proj = register_module("k_proj", torch::nn::Linear(hidden_size_, hidden_size_));
        v_proj = register_module("v_proj", torch::nn::Linear(hidden_size_, hidden_size_));
        o_proj = register_module("o_proj", torch::nn::Linear(hidden_size_, hidden_size_));
    }

    torch::Tensor
    forward(const torch::Tensor& x, const c10::optional<torch::Tensor>& mask = c10::nullopt)
    {
        auto B = x.size(0), T = x.size(1), C = x.size(2);

        auto q = q_proj->forward(x).view({B, T, num_heads, head_dim}).permute({0, 2, 1, 3});
        auto k = k_proj->forward(x).view({B, T, num_heads, head_dim}).permute({0, 2, 1, 3});
        auto v = v_proj->forward(x).view({B, T, num_heads, head_dim}).permute({0, 2, 1, 3});

        auto attn_scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

        if (mask.has_value())
            attn_scores = attn_scores.masked_fill(mask.value().eq(0), -1e9);

        auto attn_probs = torch::softmax(attn_scores, -1);
        auto attn_out =
            torch::matmul(attn_probs, v).permute({0, 2, 1, 3}).contiguous().view({B, T, C});

        return o_proj->forward(attn_out);
    }
};

// -----------------------------
// FeedForward (SwiGLU)
// -----------------------------
struct FeedForwardImpl : torch::nn::Module
{
    torch::nn::Linear fc_in{nullptr}, fc_gate{nullptr}, fc_out{nullptr};

    FeedForwardImpl(int64_t hidden_size, int64_t mult = 4)
    {
        int64_t inner = hidden_size * mult;
        fc_in = register_module(
            "fc_in", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, inner).bias(true)));
        fc_gate = register_module(
            "fc_gate", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, inner).bias(true)));
        fc_out = register_module(
            "fc_out", torch::nn::Linear(torch::nn::LinearOptions(inner, hidden_size).bias(true)));
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        return fc_out->forward(torch::silu(fc_in->forward(x)) * fc_gate->forward(x));
    }
};

// -----------------------------
// Transformer Block
// -----------------------------

TORCH_MODULE(RMSNorm);            // 产生 RMSNorm (holder) 对应 RMSNormImpl
TORCH_MODULE(MultiHeadAttention); // 产生 MultiHeadAttention 对应 MultiHeadAttentionImpl
TORCH_MODULE(FeedForward);        // 产生 FeedForward 对应 FeedForwardImpl

// 然后把 TransformerBlockImpl 改成使用 wrapper 类型作为成员：
struct TransformerBlockImpl : torch::nn::Module
{
    RMSNorm ln1{nullptr};
    RMSNorm ln2{nullptr};
    MultiHeadAttention attn{nullptr};
    FeedForward mlp{nullptr};

    TransformerBlockImpl(int64_t hidden_size, int64_t num_heads)
    {
        ln1 = register_module("ln1", RMSNorm(hidden_size));
        ln2 = register_module("ln2", RMSNorm(hidden_size));
        attn = register_module("attn", MultiHeadAttention(hidden_size, num_heads));
        mlp = register_module("mlp", FeedForward(hidden_size));
    }

    torch::Tensor
    forward(const torch::Tensor& x, const c10::optional<torch::Tensor>& mask = c10::nullopt)
    {
        auto h = x + attn->forward(ln1->forward(x), mask);
        h = h + mlp->forward(ln2->forward(h));
        return h;
    }
};
TORCH_MODULE(TransformerBlock);

// -----------------------------
// Qwen Model
// -----------------------------
struct QwenForCausalLMImpl : torch::nn::Module
{
    torch::nn::Embedding embed_tokens{nullptr}, pos_embed{nullptr};
    torch::nn::ModuleList layers{nullptr};
    RMSNorm ln_f{nullptr};
    torch::nn::Linear head{nullptr};

    QwenForCausalLMImpl(
        int64_t vocab_size,
        int64_t hidden_size,
        int64_t num_layers,
        int64_t num_heads,
        int64_t max_seq_len)
    {
        embed_tokens =
            register_module("embed_tokens", torch::nn::Embedding(vocab_size, hidden_size));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(max_seq_len, hidden_size));
        // register a ModuleList for transformer layers and append blocks to it
        layers = register_module("layers", torch::nn::ModuleList());
        for (int64_t i = 0; i < num_layers; ++i)
        {
            TransformerBlock blk(hidden_size, num_heads);
            layers->push_back(blk);
        }
        ln_f = register_module("ln_f", RMSNorm(hidden_size));
        head = register_module(
            "head",
            torch::nn::Linear(torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)));
    }

    torch::Tensor
    forward(const torch::Tensor& input_ids, const c10::optional<torch::Tensor>& mask = c10::nullopt)
    {
        auto B = input_ids.size(0);
        auto T = input_ids.size(1);
        auto device = input_ids.device();
        auto positions = torch::arange(T, torch::TensorOptions().dtype(torch::kLong).device(device))
                             .unsqueeze(0)
                             .expand({B, T});
        auto x = embed_tokens->forward(input_ids) + pos_embed->forward(positions);
        for (size_t i = 0; i < layers->size(); ++i)
        {
            // Retrieve the module holder for the transformer block using ptr()
            // and cast it to the concrete implementation type. This avoids
            // depending on a templated at<T>() member and is robust at runtime.
            auto module_holder = layers->ptr(i);
            auto tb = std::dynamic_pointer_cast<TransformerBlockImpl>(module_holder);
            if (!tb)
            {
                throw std::runtime_error(
                    "Failed to cast layers[" + std::to_string(i) + "] to TransformerBlockImpl");
            }
            x = tb->forward(x, mask);
        }
        x = ln_f->forward(x);
        return head->forward(x);
    }
};
TORCH_MODULE(QwenForCausalLM);

// -----------------------------
// Greedy Generation (C++ Module)
// -----------------------------
std::string generate_from_module(
    std::shared_ptr<QwenForCausalLMImpl> model,
    const std::string& prompt,
    int max_new_tokens = 50,
    double temperature = 1.0)
{
    model->eval();

    auto ids = encode(prompt);
    auto input_ids = torch::tensor(ids, torch::kLong).unsqueeze(0).to(DEVICE);

    for (int i = 0; i < max_new_tokens; ++i)
    {

        auto logits = model->forward(input_ids);
        auto next_logits = logits.index({0, -1}).div(temperature);
        auto next_id = next_logits.argmax(-1).unsqueeze(0).unsqueeze(0);

        std::vector<torch::Tensor> cat_in;
        cat_in.push_back(input_ids);
        cat_in.push_back(next_id);

        input_ids = torch::cat(cat_in, 1);
    }

    auto out = input_ids.squeeze(0).cpu();
    std::vector<int64_t> ids_out(out.data_ptr<int64_t>(), out.data_ptr<int64_t>() + out.numel());
    return decode(ids_out);
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv)
{

    std::cout << "Using device: " << DEVICE << std::endl;
    const std::string module_path = "<Your-QWen2-StateDict-Weights-File>";

    /* create model */
    std::shared_ptr<QwenForCausalLMImpl> model = nullptr;
    model = std::make_shared<QwenForCausalLMImpl>(
        VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN);
    model->to(DEVICE);

    /* load weights */
    if (std::ifstream(module_path))
    {
        try
        {
            torch::load(model, module_path);
            std::cout << "Loaded weights from " << module_path << std::endl;
        }
        catch (...)
        {
            std::cout << "Warning: load failed, using random weights." << std::endl;
        }
    }

    /* inference */
    std::string prompt = "你好啊";
    std::string output;
    auto t0 = std::chrono::high_resolution_clock::now();

    output = generate_from_module(model, prompt, 50);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\n生成结果:\n" << output << std::endl;
    std::cout << "耗时: " << std::chrono::duration<double>(t1 - t0).count() << " 秒\n";

    return 0;
}
