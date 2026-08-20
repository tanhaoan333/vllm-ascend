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
| 任务级 benchmark | 用户已验证 NPU 与 GPU 精度基本一致 |

因此：

> 当前 NPU vLLM-Omni 与纯模型在算法和任务质量上是一致的，但自然运行路径尚未达到逐层 bitwise EXACT。最早的直接模型分歧位于 Speaker Encoder；AudioVAE、Semantic、Aggregator、主 LLM 和 DiT 的差异随后叠加。DiT 初始噪声在两个独立进程中不一致，是首 patch 输出差异的主要放大因素之一。

### 0.2 不能将本结果解释为“模型精度差”

当前 benchmark 表明 NPU 与 GPU 的生成质量几乎一致。本报告测量的是更严格的逐 bit 数值一致性，以下差异不会自动等价为质量退化：

- 不同 NPU kernel 的 reduction/cast 顺序；
- vLLM Paged KV 与纯模型 DynamicCache；
- vLLM 的 bucket padding；
- 两侧 AudioVAE posterior 的随机调用次数；
- 两个进程中不同的 NPU RNG 消费历史；
- FIA 与纯模型 attention 路径差异。

---

## 1. 实测环境

| 项目 | 值 |
|---|---|
| Docker 容器 | `tanhaoan` |
| Conda 环境 | `hhh` |
| 物理设备 | NPU7 |
| NPU 型号 | Ascend950PR |
| 容器内逻辑设备 | `npu:0` |
| PyTorch | `2.9.0+cpu`，Ascend 定制构建 |
| torch-npu | `2.9.0.post4` |
| Transformers | `4.57.6` |
| vLLM | `0.20.1.dev336+g0f24991c7` |
| vLLM-Omni commit | `f411c7ff5e568feb164e28126d4498b663304e38` |
| 纯模型 commit | `c6a49adf16f6fd6eb2d18ef9a1c869f602f2b74a` |
| dtype | BF16；waveform、部分归一化及 solver 为 FP32 |
| batch | 1 |
| async scheduling | 关闭 |
| streaming | 关闭，`async_chunk=false` |
| ACLGraph/NPUGraph | 关闭，`enforce_eager=true`、`cudagraph_mode=NONE` |
| UniDiTAR 后加载融合 | 关闭，`VLLM_OMNI_ENABLE_UNIDITAR_NPU_FUSIONS=0` |
| prefix caching | 关闭 |
| seed | 42 |
| ODE steps | 10 |
| CFG alpha | 2.0 |
| fm_scale | 0.999 |
| sample strategy | `base` |



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

纯模型实际运行于物理 NPU7：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
python tools/uniditar_precision_hf_npu.py \
  --output /data/tha/bitwise_runs/0817_npu7_20260820_1245 \
  --model /data/tha/models \
  --seed 42
```

纯模型 `/data/tha/UniDiTAR` 的 Speaker mel 默认使用 `torch.stft`，当前 Ascend950PR runtime 明确报：

```text
STFT is not supported on this platform
```

为使纯模型能在 NPU 上执行，本次测试脚本仅在验证进程内将纯模型 mel 函数替换为 vLLM-Omni NPU 已验证的数学等价显式 DFT：

```text
gather framing + cos/sin DFT basis + matmul
```

没有修改 `/data/tha/UniDiTAR` 生产代码和权重。该处理也使 Speaker 输入特征路径更适合直接比较。

### 3.2 vLLM-Omni

使用独立验证 YAML，未覆盖生产配置：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
VLLM_OMNI_ENABLE_UNIDITAR_NPU_FUSIONS=0 \
UNIDITAR_PRECISION_DUMP_DIR=/data/tha/bitwise_runs/0817_npu7_20260820_1245 \
vllm-omni serve /data/tha/models \
  --omni \
  --served-model-name UniDiTAR \
  --deploy-config /data/tha/0817/vllm-omni/tools/uniditar_npu_bitwise_eager_b1.yaml \
  --trust-remote-code \
  --disable-log-stats \
  --host 0.0.0.0 \
  --port 8021
```

固定请求：

```bash
python tools/send_uniditar_precision_request.py \
  --port 8021 \
  --output /data/tha/bitwise_runs/0817_npu7_20260820_1245/10002287-00000095_vllm.wav
```

请求返回 HTTP 200。首 patch 解码输出：

| 项目 | 值 |
|---|---:|
| 文件字节数 | 9644 |
| 声道 | 1 |
| 采样率 | 24000 Hz |
| 样本数 | 4800 |
| 时长 | 0.2 s |
| sample width | 2 bytes |

输出文件：

- `/data/tha/bitwise_runs/0817_npu7_20260820_1245/10002287-00000095_vllm.wav`

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

完整 158 项逐层结果：

- `/data/tha/bitwise_runs/0817_npu7_20260820_1245/comparison.md`
- `/data/tha/bitwise_runs/0817_npu7_20260820_1245/comparison.json`

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

