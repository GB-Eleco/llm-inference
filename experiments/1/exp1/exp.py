import argparse
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

SENTENCES = [
    "小明喜欢吃苹果。\n",
    "小红喜欢吃香蕉。\n",
    "小明喜欢喝牛奶。\n",
    "小红喜欢喝果汁。\n",
]
CORPUS = "".join(SENTENCES) * 100
CHARS = sorted(set(CORPUS))
TO_ID = {ch: i for i, ch in enumerate(CHARS)}
TO_CHAR = {i: ch for i,ch in enumerate(CHARS)}

class CasualSelfAttention(nn.Module):
    def __init__(self, d_model=32, n_head=4):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim= d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, x):
        B, T, C = x.shape
        # 将[B, T, C]转换成[B, T, n, dn],再转置为[B,n,T,dn]
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)


    def forward(self, x, past_kv=None, inspect=False):
        B, T, C = x.shape
        q = self._heads(self.q(x))
        k_new = self._heads(self.k(x))
        v_new = self._heads(self.v(x))

        # KV Caching
        # 这里past_kv?
        past_len = 0 if past_kv is None else past_kv[0].size(2)
        if past_kv is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([past_kv[0], k_new], dim=2)
            v = torch.cat([past_kv[1], v_new], dim=2)

        # [B,H,T,T_total]
        # @是乘法
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_pos = past_len + torch.arange(T, device=x.device)
        k_pos = torch.arange(k.size(2), device=x.device)
        allowed = k_pos[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=32, n_head=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CasualSelfAttention(d_model, n_head)


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, block_size=32, d_model=32, n_head=4, n_layer=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head) for _ in range(n_layer)]
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=350)
    args=parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if(torch.cuda.is_available() == False):
        print("cuda is not available.")
        exit(1)
    else:
        device = "cuda"
    model = TinyDecoder(len(CHARS)).to(device)

