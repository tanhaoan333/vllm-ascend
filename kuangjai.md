## 1.1.1 实现接口设计

模型部分计算采用多个独立算子组合实现，存在中间张量读写和多次算子提交开销。针对 NPU 推理场景，采用 `torch_npu` 融合接口与自定义 Triton 算子，在满足输入形状、数据类型及运行条件时替换原计算路径。
以下按照 “**优化范围 — 原始实现 — 优化接口 — 使用约束**” 说明。代码为报告用接口示例，保留关键计算与参数；其中明确标注的 “计算示意” 用于说明等价关系，不代表逐字源码。当前实现使用 `torch_npu` 与自定义接口，不涉及 `mindiesd` 迁移。

### 1．FA 融合

Attention 的计算为：
\(O=\operatorname{Softmax}\left(\frac{QK^{\mathsf T}}{\sqrt d}+M\right)V\)
使用 `npu_fused_infer_attention_score` 执行融合 Attention。根据输入组织方式，分别采用 Dense 和 Paged 接口。

#### 1）Dense Attention 迁移

**优化范围：**Stage 1 的 Codec Token Encoder，包括 Acoustic Delay Transformer 和 Token Projector。
**a）原始实现**
输入采用 `BNSD` 布局，即 batch、head、sequence、head dimension。

```
```python
output = F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
)

```
```

**b）优化接口**

```
```python
import torch_npu
blocked_mask = None if attention_mask is None else ~attention_mask
if blocked_mask is not None:
    if blocked_mask.ndim == 2:
        blocked_mask = blocked_mask.unsqueeze(0)
    blocked_mask = blocked_mask.contiguous()
output = torch_npu.npu_fused_infer_attention_score(
    q.contiguous(),
    k.contiguous(),
    v.contiguous(),
    num_heads=q.shape[1],
    num_key_value_heads=k.shape[1],
    input_layout="BNSD",
    scale=q.shape[-1] ** -0.5,
    sparse_mode=0,
    atten_mask=blocked_mask,
    pre_tokens=2147483647,
    next_tokens=2147483647,
)[0]

```
```

**注意事项：**

- 当前路径支持 FP16/BF16 推理输入，要求关闭梯度。
- SDPA 的 bool mask 中 `True` 表示允许访问，FIA 的 bool mask 表示屏蔽位置，因此需要取反。
- 本示例对应显式 mask 的非 causal 调用；不能在 causal 场景中直接省略屏蔽逻辑。
- 融合范围不包含 QKV 投影和 Attention 输出投影。

#### 2）Paged Attention 迁移

**优化范围：**Stage 1 的 DiT Attention，读取当前请求对应的历史与当前 chunk KV Cache。
**a）原始接口形式**
原路径采用分页 FlashAttention 接口，传入页表和真实序列长度。

```
```python
output = flash_attn_varlen_func(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q=cu_seqlens_q,
    seqused_k=kv_lengths,
    block_table=block_table,
    causal=False,
    softmax_scale=head_dim ** -0.5,
)

```
```

**b）优化接口**
以下为非 causal paged FIA 调用形式；KV Cache 已包含允许读取的历史和当前 chunk。

```
```python
import torch_npu
num_blocks, block_size, kv_heads, head_dim = k_cache.shape
key = k_cache.reshape(num_blocks, block_size, kv_heads * head_dim)
value = v_cache.reshape(num_blocks, block_size, kv_heads * head_dim)
output = torch_npu.npu_fused_infer_attention_score(
    query=q.contiguous(),
    key=key,
    value=value,
    num_heads=q.shape[1],
    num_key_value_heads=kv_heads,
    input_layout="TND",
    scale=head_dim ** -0.5,
    block_table=block_table.to(dtype=torch.int32),
    block_size=block_size,
    actual_seq_lengths=actual_q,
    actual_seq_lengths_kv=actual_kv,
    sparse_mode=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
)[0].view_as(q)

```
```

**注意事项：**

- Q 布局为 `[total_tokens, heads, head_dim]`。
- 单层 KV Cache 布局为 `[num_blocks, 128, kv_heads, head_dim]`，当前实现要求页大小为 128。
- `actual_q` 为各请求 query 的累计结束位置；`actual_kv` 为各请求有效 KV 长度。
- 图执行路径还需保留 task capture 和动态参数更新，不能将捕获时的 KV 长度固定用于所有请求。
- 此处给出的是已核对的 Stage 1 实现，不将外部 LLM Attention 后端假定为相同接口。

---

### 2．KV Cache 联合写入

**优化范围：**Stage 1 DiT 的 paged KV Cache 写入阶段。
原实现分别写入 K 和 V，优化后使用 `npu_scatter_pa_kv_cache` 联合写入。
**a）原始实现**

```
```python
k_cache.view(-1, heads, head_dim).index_copy_(
    0, write_idx, key
)
v_cache.view(-1, heads, head_dim).index_copy_(
    0, write_idx, value
)

```
```

**b）优化接口**

```
```python
import torch_npu
torch_npu.npu_scatter_pa_kv_cache(
    key=key.contiguous(),
    value=value.contiguous(),
    key_cache=k_cache,
    value_cache=v_cache,
    slot_mapping=write_idx.contiguous(),
    cache_mode="Norm",
)

```
```

**注意事项：**

- K/V 输入为 `[tokens, heads, head_dim]`，缓存为四维分页布局。
- `write_idx` 是缓存线性地址，即页编号乘页大小再加页内偏移，不是请求编号。
- 该接口只负责缓存写入，不包含 QKV 投影或 Attention。
- 写入算子出现异常时，不能未经检查直接重试，以免对部分已写入的数据重复操作。

---

### 3．AdaLN 融合

**优化范围：**Stage 1 DiT Block 的 Attention 输入调制。
计算为：
\(y=\operatorname{LN}(x)\odot(1+\mathrm{scale})+\mathrm{shift}\)
将 LayerNorm、逐通道缩放和偏移合并为 `adaln_batched_kernel`，通过 `adaln_modulate` 接口调用。
**a）原始计算示意**

```
```python
normalized = F.layer_norm(
    x.float(),
    normalized_shape=(x.shape[-1],),
    weight=None,
    bias=None,
    eps=1e-6,
)
output = (
    normalized * (1 + scale[:, None, :])
    + shift[:, None, :]
)

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.stage1_fusions import (
    adaln_modulate,
)
output = adaln_modulate(x, shift, scale)

```
```

**注意事项：**

- 输入为 FP32 `[B,T,D]`，调制参数为 FP32 `[B,D]`。
- 当前融合采用无 affine 的 LayerNorm，`eps=1e-6`。
- 作用于有条件调制的 DiT Block，不覆盖默认无调制的 Token Encoder。
- `FinalLayer` 的原生归一化与调制不包含在本融合中。

---

### 4．门控残差与 AdaLN 融合

**优化范围：**Stage 1 DiT Block 的 Attention 输出到 FFN 输入之间。
计算包括：
\(u=x+\mathrm{gate}_a\odot\mathrm{attention}\)
\(f=\operatorname{LN}(u)\odot(1+\mathrm{scale}_f)+\mathrm{shift}_f\)
通过 `gated_residual_adaln` 接口调用 `gated_residual_adaln_kernel`。
**a）原始计算示意**

```
```python
residual_output = (
    x + gate_a[:, None, :] * attention.float()
)
normalized = F.layer_norm(
    residual_output,
    normalized_shape=(x.shape[-1],),
    weight=None,
    bias=None,
    eps=1e-6,
)
feedforward_input = (
    normalized * (1 + scale_f[:, None, :])
    + shift_f[:, None, :]
)

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.stage1_fusions import (
    gated_residual_adaln,
)
residual_output, feedforward_input = gated_residual_adaln(
    x,
    attention,
    gate_a,
    shift_f,
    scale_f,
)

```
```

**注意事项：**

- 返回顺序为 “更新后的残差、FFN 输入”，不能颠倒。
- 残差与调制参数保持 FP32，Attention 分支可使用主干计算类型。
- 融合范围不包含 FFN，也不包含 FFN 后的 `gate_f` 乘法和残差加法。

---

### 5．Q/K LayerNorm 与 RoPE 融合

**优化范围：**Stage 1 DiT Attention，以及 Token Encoder 中启用 QK Norm 的 Acoustic Delay Attention。
将每个 head 的 Q/K LayerNorm 与位置旋转合并，通过 `apply_codec_qk_norm_rope_fp32` 调用 `qk_layernorm_rope`。
**a）原始计算示意**

```
```python
q32 = F.layer_norm(
    q.float(),
    (attn.head_dim,),
    attn.q_norm.weight.float(),
    attn.q_norm.bias.float(),
    attn.q_norm.eps,
)
k32 = F.layer_norm(
    k.float(),
    (attn.head_dim,),
    attn.k_norm.weight.float(),
    attn.k_norm.bias.float(),
    attn.k_norm.eps,
)
half = attn.head_dim // 2
q_rotated = torch.cat(
    (-q32[..., half:], q32[..., :half]), dim=-1
)
k_rotated = torch.cat(
    (-k32[..., half:], k32[..., :half]), dim=-1
)
q_output = (q32 * cos + q_rotated * sin).to(q.dtype)
k_output = (k32 * cos + k_rotated * sin).to(k.dtype)

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.uniditar_rope import (
    apply_codec_qk_norm_rope_fp32,
)
q_output, k_output = apply_codec_qk_norm_rope_fp32(
    attn, q, k, cos, sin
)

```
```

**注意事项：**

- Q/K 采用 `[B,T,H,d]` 布局，V 不参与该融合。
- 当前融合条件包括 `B*T*H ≤ 2048`、每头维度为正偶数且不超过 256。
- 接口使用实际 Q/K LayerNorm 的 affine 参数和 `eps`。
- 默认未启用 QK Norm 的 Token Projector 不适用该融合；超出条件时执行原生路径。

---

### 6．RoPE 参数表索引与写入融合

**优化范围：**Stage 1 Whole-solve 的图输入准备。
将位置索引、有效性判断、cos/sin 查表和输出写入合并，通过 `stage_rope_tables` 调用 `rope_stage_kernel`。
**a）原始计算示意**

```
```python
positions = offsets[:, None] + frame_ids[None, :]
positions = positions.clamp(
    max=cos_table.shape[0] - 1
).reshape(-1)
cos_values = cos_table.index_select(
    0, positions
).view_as(cos_output)
sin_values = sin_table.index_select(
    0, positions
).view_as(sin_output)
cos_output.copy_(
    cos_values.masked_fill(~valid_frames[None, :, None], 0)
)
sin_output.copy_(
    sin_values.masked_fill(~valid_frames[None, :, None], 0)
)

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.stage1_fusions import (
    stage_rope_tables,
)
stage_rope_tables(
    (cos_table, sin_table),
    offsets,
    valid_batch,
    valid_width,
    (cos_output, sin_output),
)

```
```

**注意事项：**

- 该算子准备位置参数，不执行 Q/K 的 RoPE 旋转。
- 表为 `[R,D]`，输出为 `[B,T,D]`。
- `valid_batch`、`valid_width` 为设备端单元素整数张量。
- 融合实现对越界位置补零，原生路径存在上界截断。因此等价替换要求有效位置落在已准备的表范围内，不能宣称任意越界输入等价。

---

### 7．Snake 激活融合

**优化范围：**Stage 1 Vocoder Decoder 的 AMP 残差块与最终抗混叠激活。
SnakeBeta 的计算为：
\(y=x+\frac{\sin^2(\alpha x)}{\beta+\varepsilon}\)
其中 `exp_alpha` 和 `inv_beta` 为预计算参数。
**a）原始实现**

```
```python
output = x + inv_beta * torch.pow(
    torch.sin(x * exp_alpha), 2
)

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.uniditar_snake import (
    run_uniditar_npu_snake,
)
output = run_uniditar_npu_snake(snake_module, x)

```
```

接口内部根据形状选择：

- `snake_fp32`：完整公式由一个 SIMT kernel 执行。
- `snake_pointwise`：长序列采用前置 pointwise、原生 `Sin`、后置 pointwise 三段执行。
长序列路径的结构为：
\(x \xrightarrow{\text{pointwise}} \alpha x
\xrightarrow{\sin} s
\xrightarrow{\text{pointwise}} x+\mathrm{inv\_beta}\,s^2\)
**注意事项：**
- 当前优化针对 FP32 `[B,C,L]` 推理输入。
- `B ≤ 2` 且 `L ≥ 8192` 时选择长序列路径，并受最大长度及其他支持条件限制。
- 长序列路径没有将 `Sin` 融合到单个 kernel。
- 参数变化后需重新准备派生缓存，不能继续使用失效参数。

---

### 8．FIR 上采样与下采样特化融合

**优化范围：**Stage 1 Vocoder 的抗混叠激活。
采用固定 12-tap、2 倍采样的 `fir_fp32_vector`，通过 `run_uniditar_npu_fir` 接口调用。
**a）原始计算示意**
上采样的数学实现可表示为：

```
```python
padded = F.pad(x, (5, 5), mode="replicate")
output = 2 * F.conv_transpose1d(
    padded,
    up_filter,
    stride=2,
    groups=channels,
)
output = output[..., 15:-15]

```
```

下采样可表示为：

```
```python
padded = F.pad(
    x, (pad_left, pad_right), mode="replicate"
)
output = F.conv1d(
    padded,
    down_filter,
    stride=2,
    groups=channels,
)

```
```

上采样的当前 NPU 原生路径已采用相位卷积，以上转置卷积形式用于解释数学关系。
**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.uniditar_fir import (
    run_uniditar_npu_fir,
)
upsampled = run_uniditar_npu_fir(upsample_module, x)
downsampled = run_uniditar_npu_fir(
    downsample_module, activated
)

```
```

**注意事项：**

- 当前针对 FP32、12-tap、2 倍采样及受支持的边界配置。
- 接口内部仍单独执行 replicate padding，不能表述为消除了全部边界处理算子。
- 融合 kernel 直接读取已加载 filter。
- 下采样融合为额外启用的条件路径，默认保持原生实现。
- 上采样和下采样分别调用，不能将整个抗混叠激活视为一个 kernel。

---

### 9．残差 Add 与 LayerNorm 融合

**优化范围：**Stage 0 匹配到的 Fbank Encoder 层。
计算为：
\(r=x_1+x_2,\qquad y=\operatorname{LN}(r)\)
使用 `npu_add_layer_norm` 同时返回归一化输出与残差输出。
**a）原始实现**

```
```python
residual = left + right
normalized = norm(residual)

```
```

**b）优化接口**

```
```python
import torch_npu
result = torch_npu.npu_add_layer_norm(
    left,
    right,
    norm.weight,
    norm.bias,
    float(norm.eps),
    True,
)
normalized = result[0]
residual = result[3]

```
```

**注意事项：**

- 必须保留实际 LayerNorm 的 weight、bias 和 `eps`。
- 本实现使用返回值第 0 项作为归一化输出，第 3 项作为加法残差。
- 该算子不包含 gate 乘法，也不是 DiT 的 AdaLN。
- 是否覆盖某个参考编码器实例取决于模块安装时机，不能仅依据类名认定已启用。

---

### 10．MoE 单 Token 分发与归并特化

**优化范围：**Stage 0 LLM MoE 的 token dispatch 和 token combine。
该优化针对固定单 token 场景，用专用实现替换通用路由整理与归并，不包含 Router、专家矩阵计算或完整 MoE。

#### 1）Token Dispatch

**a）原始接口**

```
```python
result = torch_npu.npu_moe_init_routing_v2(
    hidden,
    expert_ids,
    active_num=8,
    expert_num=128,
    expert_tokens_num_type=1,
    expert_tokens_num_flag=True,
    active_expert_range=[0, 128],
    quant_mode=-1,
)
expanded, row_mapping, expert_counts = result[:3]

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.tts_codec_stage0_kernels import (
    dispatch_one,
)
expanded, row_mapping, expert_counts = dispatch_one(
    hidden, expert_ids
)

```
```

#### 2）Token Combine

**a）原始接口**

```
```python
output = torch_npu.npu_moe_token_unpermute(
    permuted_tokens=expert_output,
    sorted_indices=torch.abs(row_mapping),
    probs=topk_weights,
)

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.tts_codec_stage0_kernels import (
    combine_one,
)
output = combine_one(
    permuted_tokens=expert_output,
    sorted_indices=row_mapping,
    probs=topk_weights,
)

```
```

**注意事项：**

- 当前特化要求单 token、隐藏维度 2048、Top-K 为 8、专家数为 128，并满足 BF16 和并行布局约束。
- Combine 必须使用自有 Dispatch 生成的映射，不能接受任意上游映射。
- 实现保留相应权重舍入和累加语义，不能仅用普通加权求和假定数值行为完全相同。
- 不满足条件时保留通用路由路径。

---

### 11．Logits 转换、候选写入与屏蔽融合

**优化范围：**Stage 0 LLM 输出投影之后、采样之前。
通过 `write_logits` 合并 FP32 转换、合法候选写入和非法位置屏蔽。
**a）原始计算示意**

```
```python
mask = torch.ones(
    vocab, device=source.device, dtype=torch.bool
)
mask[lo:hi] = False
if allow_end:
    mask[end] = False
logits = source.to(
    dtype=torch.float32, copy=False
).masked_fill(mask, float("-inf"))

```
```

**b）优化接口**

```
```python
from vllm_omni.platforms.npu.ops.tts_codec_stage0_kernels import (
    write_logits,
)
logits = write_logits(
    source,
    vocab=vocab,
    lo=lo,
    hi=hi,
    end=end,
    allow_end=allow_end,
    compact=False,
)

```
```

当输入来自 compact head 时：

```
```python
logits = write_logits(
    compact_source,
    vocab=vocab,
    lo=lo,
    hi=hi,
    end=end,
    allow_end=allow_end,
    compact=True,
    end_column=end_column,
)

```
```

**注意事项：**

- `[lo, hi)` 为合法候选区间，`hi` 是排他上界。
- Compact 输入的 Codec 列与完整词表列不同，结束 token 位置由 `end_column` 指定。
- 返回新分配的 FP32 完整词表 logits，不提供外部 `out` 参数。
- 不包含 LM-head GEMM，也不执行随机采样。

---

### 12．Top-K 候选过滤融合

**优化范围：**Stage 0 采样器的候选过滤阶段。
使用 `npu_top_k_top_p` 替换支持条件下的原候选过滤调用。当前集成主要覆盖 `p=1` 的 Top-K 场景。
**a）原始接口形式**

```
```python
filtered_logits = native_filter(
    logits, k, p, top_k
)

```
```

这里的 `native_filter` 表示保存的原采样器过滤接口，非新增算子。
**b）优化接口**

```
```python
import torch_npu
p_tensor = torch.ones(
    logits.shape[0],
    device=logits.device,
    dtype=torch.float32,
)
k_tensor = k.to(
    device=logits.device,
    dtype=torch.int32,
)
filtered_logits = torch_npu.npu_top_k_top_p(
    logits,
    p_tensor,
    k_tensor,
)

```
```

**注意事项：**

- 接口位置参数顺序为 logits、P、K，不能交换。
- 当前路径使用连续 FP32 logits，要求 `p=1` 且 `1 ≤ k ≤ min(vocab, 1024)`。
- 实际实现复用预分配的全一 P buffer，示例中的创建语句用于说明输入格式。
- 仅替换候选过滤，不改变 RNG、最终采样和 logprob 路径。
- 不满足条件时执行原过滤接口。

---

### 实现约束

各融合与特化接口应根据输入布局、类型、模型配置及运行模式选择，保留未支持场景的原生路径。报告中的 “等价替换” 指**支持条件下保持既定计算语义**，不表示所有输入均可替换，也不保证浮点结果逐位相同。部署验证应分别检查数值误差、图模式行为和实际性能收益。
