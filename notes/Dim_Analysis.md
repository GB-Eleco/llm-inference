假设输入为"Thinking Machine"，d_model=4,n_head=8
```
Tokenize("Thinking Machine")=(6,7)
Embedding&Pos((6,7))=X
# X 2x4
```
$$
Q=X \times W^Q \text{（$W^Q$ 4x3x8 $Q$ 2x3x8，这里用了广播）}
$$
KV同上