<!--Copyright 2024 The HuggingFace Team. All rights reserved.

根据 Apache License, Version 2.0（“License”）授权；除非遵守该 License，否则不得使用此文件。你可以在以下地址获取该 License 的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据该 License 分发的软件均按“原样”（AS IS）提供，
不附带任何明示或暗示的担保或条件。有关该 License 下的权限和限制，请参见 License 中的具体条款。

⚠️ 请注意，该文件是 Markdown 格式，但包含我们文档构建器使用的特定语法（类似于 MDX），
因此在你的 Markdown 查看器中可能无法正确渲染。

-->
> 本文翻译自[Hugging Face-How caching works?](https://huggingface.co/docs/transformers/main/en/cache_explanation)

# KV缓存的原理

设想一下，你正在和某个人对话，但对方不会记住你之前说过什么，每次回应你时都必须从头开始。这样会很慢，也很低效，对吧？

你可以把这个类比扩展到 transformer 模型。自回归模型的生成可能会很慢，因为它一次只预测一个 token。每一次新的预测都依赖于之前的所有上下文。

为了预测第 1000 个 token，模型需要来自前 999 个 token 的信息。这些信息以 token 表示之间的矩阵乘法形式来表达。

为了预测第 1001 个 token，除了第 1000 个 token 的任何信息之外，你还需要来自前 999 个 token 的相同信息。这意味着模型必须针对每个 token 一遍又一遍地计算大量矩阵乘法！

键值（key-value，KV）缓存通过存储由先前已处理 token 的注意力层得到的 kv 对，消除了这种低效。存储的 kv 对会从缓存中取出，并在后续 token 中重复使用，从而避免重新计算。

> [!WARNING]
> 缓存应只用于**推理**。如果在训练期间启用缓存，可能会导致意外错误。

为了更好地理解缓存如何工作以及为什么有效，我们进一步看看注意力矩阵的结构。

## 注意力矩阵

对于批大小为 `b`、注意力头数量为 `h`、到目前为止的序列长度为 `T`、每个注意力头的维度为 `d_head` 的情况，**缩放点积注意力**的计算如下所示。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_{\text{head}}}} \times \text{mask} \right) V
$$

查询（`Q`）、键（`K`）和值（`V`）矩阵是由输入嵌入投影得到的，其形状为 `(b, h, T, d_head)`。

对于因果注意力，mask 会阻止模型关注未来的 token。一旦某个 token 被处理，它相对于未来 token 的表示就不会再发生变化，这意味着 $ K_{\text{past}} $ 和 $ V_{\text{past}} $ 可以被缓存，并在计算最后一个 token 的表示时重复使用。

$$
\text{Attention}(q_t, [\underbrace{k_1, k_2, \dots, k_{t-1}}_{\text{cached}}, k_{t}], [\underbrace{v_1, v_2, \dots, v_{t-1}}_{\text{cached}}, v_{t}])
$$

在推理时，你只需要最后一个 token 的 query，就可以计算用于预测下一个 token $ t+1 $ 的表示 $ x_t $。在每一步中，新的 key 和 value 向量都会被**存储**到缓存中，并**追加**到过去的 key 和 value 后面。

$$
K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t)
$$

注意力是在模型的每一层中独立计算的，缓存也是按层进行的。

请参考下表，比较缓存如何提高效率。

| 不使用缓存 | 使用缓存 |
|---|---|
| 每一步都重新计算所有之前的 `K` 和 `V` | 每一步只计算当前的 `K` 和 `V` |
| 每一步的注意力计算成本随序列长度呈**二次增长** | 每一步的注意力计算成本随序列长度呈**线性增长**（内存线性增长，但每个 token 的计算量仍然较低） |

## Cache 类

一个基本的 KV 缓存接口会接收当前 token 的 key 张量和 value 张量，并返回更新后的 `K` 和 `V` 张量。这由模型的 `forward` 方法在内部管理。

```py
new_K, new_V = cache.update(k_t, v_t, layer_idx)
attn_output = attn_layer_idx_fn(q_t, new_K, new_V)
```

当你使用 Transformers 的 [`Cache`] 类时，自注意力模块会执行几个关键步骤，以整合过去和当前的信息。

1. 注意力模块会将当前的 kv 对与缓存中存储的过去 kv 对进行拼接。这会创建形状为 `(new_tokens_length, past_kv_length + new_tokens_length)` 的注意力权重。当前和过去的 kv 对本质上会被合并起来计算注意力分数，从而确保模型能够感知先前上下文和当前输入。

2. 当迭代调用 `forward` 方法时，确保 attention mask 的形状与过去和当前 kv 对的合并长度相匹配非常关键。attention mask 的形状应为 `(batch_size, past_kv_length + new_tokens_length)`。这通常由 [`~GenerationMixin.generate`] 在内部处理，但如果你想用 [`Cache`] 实现自己的生成循环，请记住这一点！attention mask 应该包含过去和当前 token 的值。

## 缓存存储实现

缓存被组织为一个层列表，其中每一层都包含一个 key 缓存和一个 value 缓存。key 缓存和 value 缓存是形状为 `[batch_size, num_heads, seq_len, head_dim]` 的张量。

层可以有不同类型（例如 `DynamicLayer`、`StaticLayer`、`StaticSlidingWindowLayer`），这主要会改变序列长度的处理方式以及缓存的更新方式。

最简单的是 `DynamicLayer`，它会随着处理更多 token 而增长。序列长度维度（`seq_len`）会随着每个新 token 的加入而增加：

```py
cache.layers[idx].keys = torch.cat([cache.layers[idx].keys, key_states], dim=-2)
cache.layers[idx].values = torch.cat([cache.layers[idx].values, value_states], dim=-2)
```

其他层类型，例如 `StaticLayer` 和 `StaticSlidingWindowLayer`，具有在创建缓存时设定的固定序列长度。这使它们能够与 `torch.compile` 兼容。对于 `StaticSlidingWindowLayer`，当新 token 被加入时，已有 token 会被移出缓存。

下面的示例展示了如何使用 [`DynamicCache`] 创建一个生成循环。如前所述，attention mask 是过去和当前 token 值的拼接。

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from accelerate import Accelerator

device = Accelerator().device

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache(config=model.config)
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

generated_ids = inputs.input_ids
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    # 贪心采样一个下一个 token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # 为下一步生成准备输入，方法是保留尚未处理的 token；
    # 在我们的例子中，只有一个新的 token，并按照上文解释的方式扩展该新 token 的 attn mask
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
```
