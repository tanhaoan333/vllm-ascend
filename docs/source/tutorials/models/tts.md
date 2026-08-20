# UniDiTAR NPU Bitwise 一致性实测报告

本文档记录对以下两套实现进行的真实 BF16、batch=1、prefill 首步、关闭 Graph、关闭 UniDiTAR 后加载融合的数值对比：


- **测试样本**：`/data/tha/seedtts_testset1/zh/meta.lst` 第一条


---

## 0. 结论

### 0.1 总结

本次自然端到端首 patch 对比结果为：

| 项目 | 实测结论 |
|---|---|
| 输入 waveform | **EXACT** |
| acoustic frame 数 | **EXACT**，两端均为 240 |
| 自动匹配 tensor | 158 组 |
| 逐 bit EXACT | 2 组 |
| 首个可直接比较的模型数值分歧 | `speaker_raw` |
| 首 patch latent | 非 bitwise，`abs_mean=8.32816482e-01` |
| 任务级 benchmark | 已验证 NPU 与 GPU 精度基本一致 |

因此：

> 当前 NPU vLLM-Omni 与纯模型在算法和任务质量上是一致的，但自然运行路径尚未达到逐层 bitwise EXACT。最早的直接模型分歧位于 Speaker Encoder；AudioVAE、Semantic、Aggregator、主 LLM 和 DiT 的差异随后叠加。DiT 初始噪声在两个独立进程中不一致，是首 patch 输出差异的主要放大因素之一。


---

## 2. 测试输入

### 2.1 SeedTTS 样本

| 字段 | 值 |
|---|---|
| fid | `10002287-00000095` |
| language | `zh` / API 使用 `Chinese` |
| prompt text | `在此奉劝大家别乱打美白针。` |
| prompt wav | `/data/tha/seedtts_testset1/zh/prompt-wavs/10002287-00000094.wav` |
| infer text | `简单地说，这相当于惠普把消费领域市场拱手相让了。` |
| use ICL | `true` |

### 2.2 Waveform 输入数值

两侧 waveform 均为：

```text
shape = [1, 115200]
dtype = torch.float32
```

前 8 个值：

| 纯模型 | vLLM-Omni |
|---|---|
| `[-0.00518798828125, -0.004425048828125, -0.00390625, -0.005950927734375, -0.004547119140625, -0.005126953125, -0.00445556640625, -0.004119873046875]` | `[-0.00518798828125, -0.004425048828125, -0.00390625, -0.005950927734375, -0.004547119140625, -0.005126953125, -0.00445556640625, -0.004119873046875]` |

指标：

```text
exact         = True
abs_mean      = 0
abs_max       = 0
relative_mean = 0
```

### 2.3 Token 对齐口径

纯模型 dump 的 `prompt_token_ids` 为 prefix 部分 `[1, 41]`；vLLM dump 位于 runner preprocess 后，为完整 packed prompt `[94]`，且 dtype 分别为 INT64/INT32。

将 vLLM 前 41 个 token 与纯模型 prefix 对齐后，数值差为 0；但因为：

- scope 不同：prefix-only vs full prompt；
- shape 不同：41 vs 94；
- dtype 不同：INT64 vs INT32；

所以该节点不计为模型数值首分歧。第一个可直接比较的模型输出是 `speaker_raw`。

---

## 3. 实际运行方式

### 3.1 纯模型

纯模型的 Speaker mel 默认使用 `torch.stft`，当前 Ascend950PR runtime 明确报：

```text
STFT is not supported on this platform
```

为使纯模型能在 NPU 上执行，本次测试脚本仅在验证进程内将纯模型 mel 函数替换为 vLLM-Omni NPU 已验证的数学等价显式 DFT：

```text
gather framing + cos/sin DFT basis + matmul
```


### 3.2 vLLM-Omni


固定请求：


| 项目 | 值 |
|---|---:|
| 文件字节数 | 9644 |
| 声道 | 1 |
| 采样率 | 24000 Hz |
| 样本数 | 4800 |
| 时长 | 0.2 s |
| sample width | 2 bytes |


---

## 4. 误差定义与 shape 对齐

对 shape 可比较的 tensor：

\[
E_{abs}=\operatorname{mean}(|x_{vllm}-x_{pure}|)
\]

\[
E_{max}=\max(|x_{vllm}-x_{pure}|)
\]

\[
E_{rel}=\frac{E_{abs}}{\operatorname{mean}(|x_{pure}|)+10^{-30}}
\]

`exact=True` 要求：

- shape 相同；
- dtype 相同；
- `torch.equal()` 为真。

### Bucket 对齐

即使关闭 Graph，当前 UniDiTAR 的 GraphSlot eager path 仍使用静态 bucket buffer。因此：

- Semantic：纯模型真实长度 240，vLLM bucket 256；比较前取 vLLM 前 240 行；
- Aggregator：纯模型真实 patch batch 24，vLLM bucket 32；比较前取 vLLM 前 24 行；
- 主 LLM：纯模型 `[1,T,D]` 与 vLLM packed `[T,D]`，仅去除纯模型 batch=1 维；
- 首 patch：纯模型 `[1,10,64]` 与 vLLM `[10,64]`，仅去除 batch=1 维。

不会 reshape、重排或截断 feature 维。

---

## 5. 模块端到端实测汇总

| 模块/节点 | shape（对齐后） | dtype | exact | abs_mean | abs_max | relative_mean |
|---|---|---|---:|---:|---:|---:|
| Prompt waveform | `[1,115200]` | FP32 | **True** | `0` | `0` | `0` |
| Speaker raw | `[1,256]` | FP32 | False | `8.22384260e-04` | `2.51865387e-03` | `1.57660755e-03` |
| Speaker projected | `[1,1536]` | BF16 | False | `3.74971278e-04` | `1.95312500e-03` | `4.40523853e-03` |
| VAE encoder output | `[1,240,128]` | BF16 | False | `2.12574732e-02` | `4.37500000e-01` | `3.64127862e-03` |
| Acoustic frame num | `[1]` | INT32 | **True** | `0` | `0` | `0` |
| Acoustic latent raw | `[1,240,64]` | BF16 | False | `3.07773259e-02` | `4.37500000e-01` | `5.49650013e-03` |
| Acoustic latent normalized | `[1,240,64]` | FP32 | False | `5.16233547e-03` | `6.85229301e-02` | `6.70999612e-03` |
| Semantic layer 0 | `[240,1280]` | BF16 | False | `1.05030859e+00` | `4.30000000e+01` | `1.76866763e-01` |
| Semantic final | `[240,1280]` | BF16 | False | `1.06241441e+00` | `1.02656250e+01` | `8.76346550e-01` |
| Aggregator layer 0 | `[24,11,1024]` | BF16 | False | `1.93286344e-01` | `1.21875000e+00` | `8.30295467e-01` |
| Aggregator final | `[24,1,1536]` | BF16 | False | `2.95603216e-01` | `1.93750000e+00` | `8.70069771e-01` |
| LLM output all | `[94,1536]` | BF16 | False | `1.51411638e-01` | `6.98168945e+00` | `1.24557646e-01` |
| LLM hidden last | `[1,1536]` | BF16 | False | `3.84874612e-01` | `2.02636719e+00` | `6.92157758e-01` |
| Stop logits | `[1,2]` | BF16 | False | `1.27172852e+00` | `1.49267578e+00` | `3.17428397e+00` |
| DiT condition | `[1,1,1536]` | BF16 | False | `3.84874612e-01` | `2.02636719e+00` | `6.92157758e-01` |
| DiT pre-context | `[1,1,20,64]` | BF16 | False | `5.51038096e-03` | `5.46875000e-02` | `6.76754781e-03` |
| DiT noise | `[1,1,10,64]` | FP32 | False | `1.19862580e+00` | `4.45516777e+00` | `1.47074545e+00` |
| DiT layer 0 / Euler step 0 | `[2,31,1024]` | BF16 | False | `5.48660420e-02` | `1.03320312e+00` | `3.65162318e-01` |
| DiT solved | `[1,1,10,64]` | FP32 | False | `8.33342075e-01` | `3.15600204e+00` | `1.08113224e+00` |
| First new patch | `[10,64]` | FP32 | False | `8.32816482e-01` | `3.15549254e+00` | `1.08153207e+00` |


---

## 6. 具体输入/输出数值

## 6.1 Speaker Encoder

### `speaker_raw`

```text
纯模型:
[0.4217047095, -0.3857417703, -0.5986943841, 0.0986975506,
 0.0883509070, 1.2812606096, 9.7324743271, 0.0947931930]

vLLM-Omni:
[0.4213241637, -0.3857730925, -0.5988527536, 0.0983870998,
 0.0897609815, 1.2821105719, 9.7299556732, 0.0954834521]
```

```text
abs_mean      = 8.22384260e-04
abs_max       = 2.51865387e-03
relative_mean = 1.57660755e-03  (0.1577%)
```

### `speaker_projected`

```text
纯模型:
[0.1894531250, 0.0306396484, -0.1621093750, 0.1118164062,
 0.0634765625, 0.0081176758, -0.1240234375, -0.0417480469]

vLLM-Omni:
[0.1904296875, 0.0317382812, -0.1621093750, 0.1113281250,
 0.0634765625, 0.0085449219, -0.1245117188, -0.0412597656]
```

```text
abs_mean      = 3.74971278e-04
abs_max       = 1.95312500e-03
relative_mean = 4.40523853e-03  (0.4405%)
```

## 6.2 AudioVAE latent

### `acoustic_latent_raw_00`

```text
纯模型:
[7.15625, 0.91796875, -3.03125, -3.421875,
 -10.625, 6.5625, -7.5625, -2.46875]

vLLM-Omni:
[7.125, 0.94140625, -3.046875, -3.375,
 -10.6875, 6.59375, -7.53125, -2.4375]
```

```text
abs_mean      = 3.07773259e-02
abs_max       = 4.37500000e-01
relative_mean = 5.49650013e-03  (0.5497%)
```

## 6.3 Semantic

### `semantic_final_00`

```text
纯模型:
[-0.1484375, -1.4609375, -0.21875, -1.375,
  0.455078125, -0.28125, 0.65234375, 0.126953125]

vLLM-Omni:
[-0.1484375, -1.4609375, -0.22265625, -1.375,
  0.462890625, -0.283203125, 0.65234375, 0.125]
```

全 tensor 指标：

```text
abs_mean      = 1.06241441e+00
abs_max       = 1.02656250e+01
relative_mean = 8.76346550e-01
```

## 6.4 Aggregator

### `aggregator_final_00`

```text
纯模型:
[-0.0539550781, -0.3457031250, 0.7343750000, 0.0673828125,
  0.6523437500, -0.3339843750, 0.0361328125, -0.0727539062]

vLLM-Omni:
[0.0996093750, -0.4570312500, 0.5664062500, -0.1337890625,
 0.8828125000, -0.0610351562, 0.0029754639, 0.0385742188]
```

```text
abs_mean      = 2.95603216e-01
abs_max       = 1.93750000e+00
relative_mean = 8.70069771e-01
```

## 6.5 主 LLM

### `llm_hidden_last_00`

```text
纯模型:
[-0.26953125, 0.431640625, -0.21875, 1.234375,
 -0.1176757812, -1.34375, 0.75390625, -1.0703125]

vLLM-Omni:
[-0.83984375, 0.73828125, -0.240234375, 1.046875,
 -0.546875, -0.8515625, 0.1591796875, 0.134765625]
```

```text
abs_mean      = 3.84874612e-01
abs_max       = 2.02636719e+00
relative_mean = 6.92157758e-01
```

### `stop_logits_00`

```text
纯模型:     [0.70703125, 0.09423828125]
vLLM-Omni: [1.7578125, -1.3984375]
```

## 6.6 DiT

### `dit_noise_00`

```text
纯模型:
[-1.7243821621, 0.2750767469, -0.8882762790, 0.1296713650,
  0.1773615628, -0.1681668609, 0.6441475153, 1.3264764547]

vLLM-Omni:
[0.3515823483, -1.4171410799, -0.8348867893, 0.3095704913,
 -0.8153618574, 0.7267753482, -0.4442059398, 0.4116063714]
```

```text
abs_mean      = 1.19862580e+00
abs_max       = 4.45516777e+00
relative_mean = 1.47074545e+00
```

### `dit_solved_00`

```text
纯模型:
[-1.1308763027, 0.3084934950, -0.3360301852, -0.0353066623,
  0.8751154542, -0.4382474422, 0.1229072809, 0.2629998922]

vLLM-Omni:
[-0.1718551368, -1.1217310429, -1.1072393656, -0.8623045087,
 -0.6432698965, 1.5433986187, -0.7558378577, -0.5797131062]
```

### 最终 `first_new_patch`

```text
纯模型:
[-1.1297454834, 0.3081850111, -0.3356941640, -0.0352713577,
  0.8742403388, -0.4378091991, 0.1227843761, 0.2627368867]

vLLM-Omni:
[-0.171875, -1.125, -1.109375, -0.86328125,
 -0.64453125, 1.546875, -0.75390625, -0.578125]
```

```text
abs_mean      = 8.32816482e-01
abs_max       = 3.15549254e+00
relative_mean = 1.08153207e+00
```

---

## 7. 分歧来源分析

## 7.1 Speaker Encoder：首个直接数值分歧

纯模型原始 mel 使用 STFT，但该算子在当前 NPU 不支持。验证脚本将其规范化为与 vLLM 相同的显式 DFT后，Speaker raw 相对误差为 0.1577%，说明：

- 输入 waveform 已 EXACT；
- mel 数学路径已统一；
- 剩余差异来自 Conformer/Linear/Norm 等实现和 BF16/NPU kernel 舍入。

这是当前自然流水线的第一个可直接比较的模型分歧。

## 7.2 AudioVAE：网络误差与 posterior 随机性混合

VAE encoder 输出相对误差为 0.3641%，raw latent 相对误差为 0.5497%。这里同时包含：

1. 24 层 Qwen2 encoder 的算子差异；
2. posterior `sample()` 的 NPU RNG状态差异。

两侧虽然都设置 seed=42，但它们是独立进程，且模型初始化、warmup、请求路径消耗随机数的顺序不同，所以 `eps~N(0,1)` 不保证相同。

严格隔离 AudioVAE 网络精度时，应该分别比较：

- posterior mean；
- posterior std；
- 使用同一份 CPU-generated epsilon 的 latent。

## 7.3 Semantic：纯模型和 vLLM 的调用图不同

纯模型实际推理路径会：

1. `encode_audio(wav)` 调用一次 AudioVAE posterior sample；
2. `encode_aggregation_input(wav)` 再次从 waveform 编码并调用 posterior sample；
3. 第二份 acoustic latent 进入 Whisper semantic。

当前 vLLM `_prompt_encode()` 只做一次 AudioVAE sample，并让 acoustic 和 semantic 共用该 latent。

因此，Semantic 的输入在自然路径上不是同一个 tensor。这解释了 Semantic 层误差快速放大，并进一步污染 Aggregator 和 LLM。

这属于**执行图/随机调用次数差异**，不能只归因于 NPU FIA 精度。

## 7.4 Bucket padding

vLLM 即使处于 eager，也复用 GraphSlot 静态 buffer：

- Semantic 240→256；
- Aggregator 24→32。

报告已经按真实长度裁剪 padding。padding 本身不是误差来源，但若不裁剪会导致 shape 无法比较。

## 7.5 主 LLM

LLM 的输入已受到 Speaker、Semantic、Aggregator 差异影响，因此本次 `llm_hidden_last` 的 69.2% 相对误差是自然端到端传播结果，不等价于主 LLM 单独算子误差。

要测主 LLM 纯算子精度，需要向两侧注入完全相同的 `inputs_embeds`，再逐层比较。

## 7.6 DiT：初始 noise 不同，当前 solved diff不能代表 DiT op精度

两侧 DiT noise 相对误差为 147.1%，已经是完全不同的随机样本。再叠加 condition/pre-context 差异后，最终 patch 非 exact 是必然结果。

严格 DiT op 对齐必须同时注入：

- 相同 condition；
- 相同 pre-context；
- 相同 FP32 noise；
- 相同 times/alpha schedule；
- 相同 solver step 数。

当前 `dit_solved` 和 `first_new_patch` 指标只能说明自然运行路径不同，不能说明 MingDiT 本身精度差。

---

## 8. 与 GPU Bitwise 报告的对应关系

GPU 报告中，各模块通过以下方法达到 EXACT：

- fixed epsilon；
- reference input injection；
- CPU 构造 RoPE cache；
- 统一 RMSNorm/RoPE/Attention 实现；
- 固定 DiT noise；
- 模块入口使用干净 HF dump。

NPU 当前自然运行结果与 GPU 报告“未注入前”的含义相同。若要求 NPU 也达到模块级 EXACT，下一步应使用本次真实 dump 继续做以下注入矩阵：

| 实验 | 注入内容 | 验证目标 |
|---|---|---|
| A | fixed posterior epsilon | AudioVAE encoder网络误差 |
| B | pure acoustic latent | Whisper Semantic |
| C | pure semantic normalized | Aggregator |
| D | pure aggregator output / inputs_embeds | 主 LLM |
| E | pure hidden_last | stop head 与 DiT condition |
| F | pure condition + pre-context + noise | MingDiT/solver |

验收时应区分：

- natural end-to-end；
- module-clean-input；
- op-clean-input；
- task benchmark。

---

## 9. 当前 Bitwise 判定

| 模块 | 自然路径 bitwise | 当前证据 |
|---|---|---|
| 输入 waveform | 通过 | `torch.equal=True` |
| Frame count | 通过 | 两端均为 240 |
| Speaker Encoder | 未通过 | 首个数值分歧，相对误差 0.1577% |
| AudioVAE Encoder | 未通过 | 网络和 posterior RNG均有差异 |
| Semantic | 未通过 | 输入 latent和调用次数不同 |
| Aggregator | 未通过 | 上游 Semantic差异传播 |
| 主 LLM | 未通过 | 上游 inputs_embeds不同，Paged KV实现不同 |
| Stop head | 未通过 | hidden_last不同 |
| MingDiT | 未通过 | condition、pre-context、noise均不同 |
| First patch | 未通过 | relative mean 108.15% |
| 任务级质量 | 基本一致 |  benchmark实测 |

最终结论：

> **当前 UniDiTAR NPU vLLM-Omni 与纯模型在 B1/BF16/eager/无Graph/无后加载融合的自然首 patch 路径下未达到 bitwise EXACT。输入 waveform和帧数完全一致；首个模型数值分歧位于 Speaker Encoder。后续差异主要由纯模型重复 posterior sample、两端独立 RNG、不同 attention/cache执行图和上游误差传播造成。**

---

## 10. Speaker Encoder 输出注入实验

### 10.1 实验方法

为判断修正 Speaker Encoder 后下游模块是否正确，本次在同一 NPU7、同一请求和同一 Eager 配置下执行第二轮实验：

1. 从纯模型基线加载 `speaker_raw.pt`；
2. 在 vLLM-Omni 的 `speaker_encoder.forward()` 出口精确替换；
3. 其余输入、权重、seed、BF16、B1、无 Graph、无后加载融合条件不变；
4. 重新比较全部 158 个匹配节点。

环境变量：

```bash
UNIDITAR_PRECISION_INJECT=speaker_raw
UNIDITAR_PRECISION_HF_DIR=/data/tha/bitwise_runs/0817_npu7_20260820_1245/hf
```

日志证据：

```text
UniDiTAR precision dumps enabled at .../vllm; injections=['speaker_raw']
POST /v1/audio/speech HTTP/1.1 200 OK
```

实验产物：

- `/data/tha/bitwise_runs/0817_npu7_speaker_inject_20260820_1416/`
- `comparison.md`
- `comparison.json`
- `server.log`

### 10.2 Speaker 分支结果

| 节点 | 自然基线 abs_mean | 注入后 abs_mean | 注入后 exact |
|---|---:|---:|---:|
| `speaker_raw` | `8.22384260e-04` | `0` | **True** |
| `speaker_projected` | `3.74971278e-04` | `0` | **True** |

结论：

- 纯模型 `speaker_raw` 已成功精确接入；
- `spkr_proj` 在相同输入下逐 bit EXACT；
- 所以 Speaker Encoder 后面的 speaker projection 模块本身正确；
- exact tensor 数由 2 增加到 4，新增项正是 `speaker_raw` 和 `speaker_projected`。

### 10.3 下一个差异模块

Speaker 注入后，AudioVAE 分支误差完全不变：

| 节点 | 自然基线 abs_mean | 注入后 abs_mean | 注入后 relative_mean |
|---|---:|---:|---:|
| `vae_encoder_output_00` | `2.12574732e-02` | `2.12574732e-02` | `3.64127862e-03` |
| `acoustic_latent_raw_00` | `3.07773259e-02` | `3.07773259e-02` | `5.49650013e-03` |
| `prompt_acoustic_normalized` | `5.16233547e-03` | `5.16233547e-03` | `6.70999612e-03` |

因此：

> **正确接入 Speaker Encoder 输出后，后续全链路仍未对齐；下一个独立产生差异的模块是 AudioVAE Encoder，而不是 `spkr_proj`。**

原因是 Speaker 和 AudioVAE 是并行分支：

```text
prompt waveform
  ├─► Speaker Encoder ─► spkr_proj ─► LLM speaker token
  └─► AudioVAE Encoder ─► acoustic latent ─► Semantic ─► Aggregator
```

修正 Speaker 不会改变 AudioVAE/Semantic 分支。

### 10.4 下游模块变化

| 节点 | 自然基线 abs_mean | Speaker注入后 abs_mean | 结论 |
|---|---:|---:|---|
| `semantic_layer_00_00` | `1.05030859e+00` | `1.05030859e+00` | 完全不变 |
| `semantic_final_00` | `1.06241441e+00` | `1.06241441e+00` | 完全不变 |
| `aggregator_final_00` | `2.95603216e-01` | `2.95603216e-01` | 完全不变 |
| `llm_output_all_00` | `1.51411638e-01` | `1.51221663e-01` | 略有改善但仍未对齐 |
| `llm_hidden_last_00` | `3.84874612e-01` | `3.86591285e-01` | 仍未对齐 |
| `stop_logits_00` | `1.27172852e+00` | `1.27954102e+00` | 仍未对齐 |
| `dit_noise_00` | `1.19862580e+00` | `1.19862580e+00` | 完全不变 |
| `dit_solved_00` | `8.33342075e-01` | `8.32571626e-01` | 仍未对齐 |
| `first_new_patch` | `8.32816482e-01` | `8.32165837e-01` | 仍未对齐 |

Semantic、Aggregator 和 DiT noise 的误差完全不变，证明它们的主要差异与 Speaker 分支无关。

### 10.5 定位结论与下一步

当前定位链为：

```text
Speaker Encoder
  ├─ 注入后 speaker_raw EXACT
  └─ spkr_proj EXACT，确认正确

AudioVAE Encoder
  └─ 下一处真实差异
       encoder output relative_mean    = 0.3641%
       raw latent relative_mean        = 0.5497%
       normalized latent relative_mean = 0.6710%

Semantic / Aggregator / LLM / DiT
  └─ 继续受 AudioVAE 分支和独立 RNG 影响
```

下一步应固定 AudioVAE posterior epsilon，分别判断：

1. 24 层 AudioVAE Encoder 网络是否一致；
2. posterior mean/std 是否一致；
3. 仅随机 epsilon 是否导致 latent 差异；
4. 注入 acoustic latent 后，Semantic 的首个真实差异位置。

---

## 11. AudioVAE Encoder 逐 Op 定位与注入

### 11.1 首个具体差异 Op

在相同 waveform 下增加以下检查点：

```text
frames → fc1 → pre_qwen2_ln → fc2 → residual
→ layer0 input RMSNorm → Attention → post RMSNorm → MLP
```

真实结果：

| 节点 | exact | abs_mean | abs_max | relative_mean |
|---|---:|---:|---:|---:|
| `vae_fc1_00` | True | `0` | `0` | `0` |
| `vae_pre_qwen2_ln_00` | True | `0` | `0` | `0` |
| `vae_fc2_00` | True | `0` | `0` | `0` |
| `vae_layer_00_input_rmsnorm_00` | False | `3.23905879e-05` | `3.90625000e-03` | `1.40791598e-03` |
| `vae_layer_00_attention_out_00` | False | `6.08599476e-05` | `1.25000000e-01` | `3.84678781e-04` |
| `vae_layer_00_output_00` | False | `8.84036708e-04` | `2.50000000e-01` | `4.74132300e-04` |

因此 AudioVAE Qwen2 的首个具体差异是：

> **第 0 层 input RMSNorm。纯模型使用 Transformers Qwen2 RMSNorm 分步 FP32 reduction；vLLM NPU 使用 `torch_npu.npu_rms_norm`。两者 reduction/cast 顺序不同。**

### 11.2 正确输出累计替换

实验将以下纯模型输出逐项替换至 vLLM：

- 24 层 input RMSNorm；
- Attention output；
- post-attention RMSNorm；
- MLP output；
- 每层 block output；
- backbone final norm；
- `fc3`；
- posterior acoustic sample；
- normalized acoustic latent。

结果：

| 节点 | 注入后结果 |
|---|---:|
| 24 层所有已记录 Op | **EXACT** |
| `vae_layer_23_output_00` | **EXACT** |
| `vae_backbone_final_norm_00` | **EXACT** |
| `vae_fc3_00` | **EXACT** |
| `vae_encoder_output_00` | **EXACT** |
| `acoustic_latent_raw_00` | **EXACT** |
| `prompt_acoustic_normalized` | **EXACT**，额外注入后消除 `1.47e-08` 的 sqrt/除法舍入 |

实验目录：

- `/data/tha/bitwise_runs/0817_npu7_vae_inject_20260820_1434/`

结论：AudioVAE 下游可继续对齐；其网络首差异由 NPU RMSNorm 开始，posterior RNG 是另一独立差异源。

---

## 12. Whisper Semantic 逐 Op 定位与修正

### 12.1 纯模型参考环境缺少 torchtune

定位过程中发现容器内：

```text
_TORCHTUNE_AVAILABLE = False
ModuleNotFoundError: No module named 'torchtune'
```

纯模型因此静默设置：

```text
rotary_embed = None
```

这不是正确 HF-reference 路径。验证脚本补入了 torchtune 等价 interleaved RoPE：

- CPU FP32 生成 theta/cos/sin cache；
- cache 搬到 NPU不改变 bit；
- `x.float()` 上执行逐项 mul/sub/add；
- 最终 cast 回 BF16。

此修正仅存在于验证脚本，没有修改纯模型仓库。

### 12.2 Semantic 前处理首差异

在注入纯模型第二次 AudioVAE sample，即 `acoustic_latent_raw_01` 后，输入已 `torch.equal=True`。但 vLLM 使用 256 bucket，纯模型使用真实 T=240，导致同一 Linear 在不同 M 维 tiling 下产生差异：

| 节点 | abs_mean | abs_max | relative_mean |
|---|---:|---:|---:|
| `semantic_pre_fc1_00` | `9.57778376e-03` | `1.0` | `1.29133239e-03` |
| `semantic_pre_gelu_00` | `4.68040118e-03` | `1.0` | `1.28894889e-03` |
| `semantic_pre_fc2_00` | `2.13330165e-02` | `2.5e-01` | `2.65754871e-03` |

将三处正确输出注入后，它们全部 EXACT，且第 0 层 `attn_ln` 也 EXACT。

### 12.3 首个 Attention 内部差异：RoPE

在相同 LayerNorm 和相同 Q/K/V 投影下：

| 节点 | exact | abs_mean | abs_max | relative_mean |
|---|---:|---:|---:|---:|
| Q pre-RoPE | True | `0` | `0` | `0` |
| K pre-RoPE | True | `0` | `0` | `0` |
| V | True | `0` | `0` | `0` |
| Q post-RoPE | False | `1.27340003e-03` | `1.25e-01` | `9.73854278e-04` |
| K post-RoPE | False | `1.23863958e-03` | `6.25e-02` | `9.89396494e-04` |
| Attention output | False | `1.91501540e-03` | `2.0` | `4.04590111e-04` |

具体原因与 GPU报告一致：

- 纯参考：torchtune 逐 Op FP32 RoPE；
- vLLM：`_DiffusionRotaryEmbedding` 平台实现；
- cache 构造设备和 kernel 中间舍入不同。

### 12.4 注入正确 RoPE 后

只注入每层 Q/K 的正确 post-RoPE，保持 NPU FIA Attention 不变：

| 节点 | 结果 |
|---|---:|
| Q/K post-RoPE | **EXACT** |
| 第 0 层 Attention output | **EXACT** |
| 第 0 层 MLP output | **EXACT** |
| 第 0 层 block output | **EXACT** |
| 第 31 层 block output | **EXACT** |
| `semantic_final_00` | **EXACT** |

这证明：

> **在 B1 prefill 条件下，正确 RoPE 输入接入后，当前 NPU FIA 与纯模型 Attention 输出可以达到 bitwise EXACT；Semantic 的核心差异来自前处理 bucket GEMM和RoPE，而不是 FIA本身。**

实验目录：

- `/data/tha/bitwise_runs/0817_npu7_rope_inject_20260820_1535/`

---

## 13. Aggregator 定位与注入

在 Semantic 32 层全部 EXACT 后，Aggregator 首个差异位于第 0 层 `norm1`：

| 节点 | abs_mean | abs_max | relative_mean |
|---|---:|---:|---:|
| `aggregator_layer_00_norm1_00` | `3.04514259e-07` | `7.8125e-03` | `3.79918860e-07` |
| `aggregator_layer_00_attention_out_00` | `1.75193315e-06` | `1.953125e-03` | `1.72994943e-05` |
| `aggregator_layer_00_output_00` | `9.82590427e-06` | `3.90625e-03` | `4.69642207e-05` |
| `aggregator_final_00` | `1.28755847e-03` | `3.125e-02` | `3.75738247e-03` |

首差异 Op 同样是：

```text
NPU npu_rms_norm vs 纯模型分步 RMSNorm
```

注入 8 层 norm1/Attention/norm2/MLP/block和 Aggregator final 后：

- 8 层全部 EXACT；
- `aggregator_final_00` EXACT。

实验目录：

- `/data/tha/bitwise_runs/0817_npu7_aggregator_optrace_20260820_1540/`

---

## 14. 主 LLM、Stop Head 与 DiT Clean-input

### 14.1 主 LLM

vLLM Qwen2 每层内部返回 `(hidden_states, residual)` 分解，而纯 Transformers Qwen2 hook看到的是合并后的 hidden。两者内部原始表示不能直接用同一个单 tensor语义比较。

验证分别完成了：

1. 注入每层纯模型 hidden，确认对应层输出检查点可以 EXACT；
2. 在 `make_omni_output` 前注入完整 `llm_output_all`，绕开 residual分解和 final norm协议差异。

结果：

| 节点 | 结果 |
|---|---:|
| `llm_output_all_00` | **EXACT** |
| `llm_hidden_last_00` | **EXACT** |
| `stop_logits_00` | **EXACT** |
| `dit_condition_00` | **EXACT** |

Stop Head 在相同 hidden 下无需任何替换即可 EXACT，说明其 Linear 实现正确。

### 14.2 DiT Clean-input

在以下输入全部 EXACT 后：

- condition；
- pre-context；
- FP32 noise；
- times/alpha schedule；
- solver steps；

DiT 仍有小误差：

| 节点 | abs_mean | abs_max | relative_mean |
|---|---:|---:|---:|
| `dit_layer_00_norm1_00` | `4.91977960e-04` | `3.125e-02` | `6.08990647e-04` |
| `dit_layer_00_attention_out_00` | `8.63841342e-05` | `7.8125e-03` | `6.76737650e-04` |
| `dit_layer_00_output_00` | `3.50312475e-04` | `1.5625e-02` | `2.32989068e-03` |
| `dit_layer_07_output_00` | `1.58124208e-03` | `1.25e-01` | `6.50572086e-03` |
| `dit_solved_00` | `6.30022585e-03` | `2.63942629e-02` | `8.10148663e-03` |
| `first_new_patch` | `6.38788054e-03` | `2.63577793e-02` | `8.22242457e-03` |

首差异再次位于第 0 层 `norm1`，即 NPU RMSNorm。

---

## 15. 最终累计 Bitwise EXACT 实验

最终实验将所有已经定位的差异边界依次替换为纯模型正确数据：

```text
speaker_raw
→ AudioVAE 24层Qwen各Op
→ acoustic raw / normalized latent
→ Semantic第二次acoustic输入
→ Semantic fc1/GELU/fc2
→ Semantic Q/K RoPE
→ Aggregator 8层各Op
→ LLM final hidden
→ DiT noise
→ DiT 8层各Op
→ DiT solved
→ first patch
```

最终核心链路结果：

| 节点 | exact | abs_mean | abs_max |
|---|---:|---:|---:|
| `speaker_raw` | True | `0` | `0` |
| `speaker_projected` | True | `0` | `0` |
| `vae_encoder_output_00` | True | `0` | `0` |
| `acoustic_latent_raw_00` | True | `0` | `0` |
| `prompt_acoustic_normalized` | True | `0` | `0` |
| `semantic_final_00` | True | `0` | `0` |
| `aggregator_final_00` | True | `0` | `0` |
| `llm_output_all_00` | True | `0` | `0` |
| `llm_hidden_last_00` | True | `0` | `0` |
| `stop_logits_00` | True | `0` | `0` |
| `dit_condition_00` | True | `0` | `0` |
| `dit_pre_context_00` | True | `0` | `0` |
| `dit_noise_00` | True | `0` | `0` |
| `dit_layer_00_norm1_00` | True | `0` | `0` |
| `dit_layer_07_08` | True | `0` | `0` |
| `dit_solved_00` | True | `0` | `0` |
| `first_new_patch` | True | `0` | `0` |

最终实验自动匹配 895 个节点，其中 866 个直接 EXACT。剩余节点是：

- `prompt_token_ids`：prefix-only INT64 与 full-packed INT32 的scope/dtype差异，数值前缀差为0；
- 28个 vLLM Qwen layer raw hook：vLLM `(hidden,residual)`内部协议与纯模型合并hidden语义不同。

这些辅助表示不影响最终功能链；`llm_output_all`、`hidden_last`、Stop Head、DiT和首 patch已全部 EXACT。


