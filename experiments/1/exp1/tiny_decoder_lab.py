"""从零学习 Decoder-only Transformer：训练、Attention、逐字生成、KV Cache。

用法：
    python tiny_decoder_lab.py
    python tiny_decoder_lab.py --steps 500
    python tiny_decoder_lab.py --steps 300 --device cpu

只依赖 PyTorch。教学模型使用字符级 Tokenizer，不追求语言能力或生成质量。
"""

import argparse
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


# 重复的短句只是为了让极小模型能很快拟合。它不是大模型的真实训练语料。
SENTENCES = [
    "小明喜欢吃苹果。\n",
    "小红喜欢吃香蕉。\n",
    "小明喜欢喝牛奶。\n",
    "小红喜欢喝果汁。\n",
]
CORPUS = "".join(SENTENCES) * 100
CHARS = sorted(set(CORPUS))
TO_ID = {ch: i for i, ch in enumerate(CHARS)}
TO_CHAR = {i: ch for ch, i in TO_ID.items()}


def encode(s):
    return [TO_ID[c] for c in s]


def decode(ids):
    return "".join(TO_CHAR[int(i)] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=32, n_head=4):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, x):
        # [B,T,C] -> [B,H,T,D]
        B, T, C = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x, past_kv=None, inspect=False):
        B, T, C = x.shape
        q = self._heads(self.q(x))
        k_new = self._heads(self.k(x))
        v_new = self._heads(self.v(x))

        # 历史 K/V 直接复用，不再为历史 Token 重算投影。
        past_len = 0 if past_kv is None else past_kv[0].size(2)
        if past_kv is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([past_kv[0], k_new], dim=2)
            v = torch.cat([past_kv[1], v_new], dim=2)

        # [B,H,T,T_total]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 允许第 (past_len+i) 个 Query 关注 Key 位置 <= past_len+i。
        q_pos = past_len + torch.arange(T, device=x.device)
        k_pos = torch.arange(k.size(2), device=x.device)
        allowed = k_pos[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = weights @ v  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        # KV 的返回不需要梯度缓存：本实验仅在 no_grad 推理时复用 KV。
        attention_to_inspect = weights.detach() if inspect else None
        return out, (k, v), attention_to_inspect


class TransformerBlock(nn.Module):
    def __init__(self, d_model=32, n_head=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, past_kv=None, inspect=False):
        attn_out, kv, weights = self.attn(self.norm1(x), past_kv, inspect)
        x = x + attn_out            # 残差连接 1
        x = x + self.mlp(self.norm2(x))  # 残差连接 2
        return x, kv, weights


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, block_size=32, d_model=32, n_head=4, n_layer=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 为便于入门，使用可学习的绝对位置 Embedding，而不是 RoPE。
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False, inspect=False):
        B, T = idx.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        if past_len + T > self.block_size:
            raise ValueError("序列超过 block_size；本教学例子不实现滑动窗口。")
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)

        new_kvs = []
        attention = None
        for i, block in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, kv, weights = block(x, past, inspect=inspect and i == 0)
            if use_cache:
                new_kvs.append(kv)
            if weights is not None:
                attention = weights

        logits = self.lm_head(self.norm(x))  # [B,T,vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss, (new_kvs if use_cache else None), attention


@torch.no_grad()
def generate_no_cache(model, prompt, max_new_tokens, device):
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _, _, _ = model(idx)   # 每一步都重复计算整段前缀
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
    return decode(idx[0].tolist())


@torch.no_grad()
def generate_with_cache(model, prompt, max_new_tokens, device):
    full = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    cache = None
    current_input = full
    for _ in range(max_new_tokens):
        logits, _, cache, _ = model(current_input, past_kvs=cache, use_cache=True)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        full = torch.cat((full, nxt), dim=1)
        current_input = nxt         # 只计算一个新 Token
    return decode(full[0].tolist())


def get_batch(data, batch_size, seq_len, device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in starts.tolist()])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    model = TinyDecoder(len(CHARS)).to(device)
    data = torch.tensor(encode(CORPUS), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print(f"device={device}, vocab_size={len(CHARS)}, params={sum(p.numel() for p in model.parameters()):,}")
    print("词表（字符级）:", CHARS)
    print("训练时输入 / 目标错位一个字符：", repr(CORPUS[:12]), "->", repr(CORPUS[1:13]))
    print("训练前:", repr(generate_no_cache(model, "小明喜欢吃", 7, device)))
    model.train()
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        x, y = get_batch(data, batch_size=32, seq_len=24, device=device)
        logits, loss, _, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = model.blocks[0].attn.q.weight.grad.norm().item()
        optimizer.step()
        if step == 0 or step % 50 == 0 or step == args.steps:
            print(f"step={step:3d}  loss={loss.item():.4f}  ||grad(W_Q)||={grad_norm:.4f}")
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"训练耗时: {time.perf_counter() - t0:.2f} 秒")

    model.eval()
    prompt = "小明喜欢吃"
    max_new_tokens = 9
    out_plain = generate_no_cache(model, prompt, max_new_tokens, device)
    out_cache = generate_with_cache(model, prompt, max_new_tokens, device)
    print("\n无 Cache 生成:", repr(out_plain))
    print("有 Cache 生成:", repr(out_cache))
    print("两种实现是否逐字一致:", out_plain == out_cache)

    with torch.no_grad():
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        logits, _, _, attention = model(idx, inspect=True)
        # 第一层第一个 Head 的最后一行：最后一个 Token 对各历史位置的注意力。
        scores = attention[0, 0, -1].cpu().tolist()
        print("\n第一层、Head 0、最后一个位置的 Attention:")
        for ch, score in zip(prompt, scores):
            print(f"  {ch}: {score:.3f}")
        probs = logits[0, -1].softmax(dim=-1)
        val, ids = probs.topk(3)
        print("下一个字符 Top-3:", [(TO_CHAR[i.item()], round(p.item(), 3)) for p, i in zip(val, ids)])

    print("\n说明：这是对极少量重复句子的拟合，不代表模型具备泛化语言能力。")
    print("后续实验：尝试改层数、Head 数、序列长度，并比较 Decode 的计算开销。")


if __name__ == "__main__":
    main()
