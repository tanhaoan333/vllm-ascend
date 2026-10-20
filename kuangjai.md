上一版确实混入了**正确性保障、资源管理和常规执行机制**，不宜全部列入性能优化报告。另外，“LLM 与双流执行”将**模块名称**和**优化技术**并列，分类标准不一致，应当调整。
建议统一按**优化作用的模块**划分：

1. Stage 0：LLM。
2. Stage 1：Codec Token Encoder。
3. Stage 1：DiT 与 Solver。
4. Stage 1：Vocoder。
5. Host：输入准备、阶段交付与输出传输。
筛选标准为：**直接减少计算量、Host/Device 交互、算子提交开销，或提供计算与传输重叠。**

## 一、建议从主要性能优化清单移除的内容

| 内容 | 处理方式 | 原因 |
| --- | --- | --- |
| 图输出 clone、事件生命周期保护、dummy slot 隔离 | 移至正确性说明 | 主要保障数据安全，并非独立性能收益。 |
| 终包去重、请求状态释放 | 移至稳定性或资源管理说明 | 属于生命周期管理。 |
| CFG=0 分支裁减 | 列入测试配置 | 当前配置不执行额外分支，不宜单独计为新增优化。 |
| 后台收发线程 | 列入框架基础能力 | 未确认该部分为本次模型适配新增。 |
| 时间频率、mask、统计量等小型缓存 | 合并为预计算，或省略 | 不宜拆成大量条目；没有测量证据时，不作为主要优化。 |
| Padding masking、真实长度控制 | 随图优化说明 | 是分桶执行的正确性条件，不单独计数。 |
| 跳过无关 outer graph 更新 | 合并进自有图执行优化 | 避免与图执行优化重复计数。 |
| 默认关闭的卷积分段 | 暂不列入成果 | 未确认实际启用和收益。 |
| CPU 输出索引适配 | 暂不单列 | 与当前 Talker 已有 Host 行号推导存在重合。 |

## 二、精简后的报告表格

### 1. Stage 0：LLM

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **共享专家与路由专家双流重叠** | LLM MoE | 将共享专家计算提交至辅助流，为两条专家路径并发执行提供条件，降低串行执行时间。 | `with torch.npu.stream(self._stream):`   　`output = self._layer(shared_experts_input)`   `torch.npu.current_stream().wait_stream(self._stream)` |
| **候选词表裁减投影** | LLM 输出头 | 仅投影合法 Codec 与结束 token 对应的权重行，减少完整词表投影中的无效计算。 | `packed = torch.cat(rows, dim=0).contiguous()`   …   `compact.register_buffer("weight", packed, persistent=False)`   …   `compact_logits = talker.logits_processor(self.head, hidden.contiguous())` |
| **LLM 图执行及动态参数更新** | LLM Runner、Attention | 复用 Ascend 图执行能力，降低重复提交开销；通过动态更新适配变化的 Attention 元数据。属于框架接入能力。 | `graph_size = NPUModelRunner.capture_model(self)`   …   `self._update_full_graph_params_if_needed(`   　`forward_context, num_tokens_padded, positions,`   `)` |
| **说明：LLM 参数更新是图执行的支撑机制，不应另行统计为一项独立加速收益。其内部是否采用与 DiT 相同的双流实现，当前证据不足。** |  |  |  |

### 2. Stage 1：Codec Token Encoder

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **Token 输入异步搬运与缓冲复用** | Token Encoder 输入准备 | 使用 pinned buffer 和非阻塞 H2D，减少输入搬运暴露的 Host 等待；通过事件查询复用已完成缓冲区。 | `ready = candidate[2].query()`   …   `host = torch.empty((batch, width), dtype=torch.long, device="cpu", pin_memory=True)`   …   `target_view.copy_(host_view, non_blocking=True)` |
| **精确形状编码图与等长批处理** | Token Encoder | 相同实际输入长度组成 batch，并重放对应编码图，减少逐请求调用及 eager 提交开销。 | `groups.setdefault((feed, session.device, session.dtype), []).append((session, keep))`   …   `key = (int(ids.shape[0]), int(ids.shape[1]))`   …   `result = slot.run(key, lambda: self.encode_tokens(buffers)).clone()` |

### 3. Stage 1：DiT 与 Solver

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **整窗 ODE 图执行与分桶** | Solver、多次 DiT 前向及 Euler 更新 | 将一个窗口的多步求解纳入整图，减少逐步 Host 调度和设备提交；按实际形状选择执行桶。 | `inputs = self._solve_scratch.view(batch, width, frames, self.z_dim)`   …   `out, histories, raw = self._solve_slot.run(key, lambda: self._solve_body(key))` |
| **FIA 参数更新与图重放协同** | DiT paged Attention | 使用独立更新流及事件协调图重放和 FIA 重绑，支持 KV 长度、页表动态变化，避免重新捕获求解图。 | `entry.npu_update_stream.wait_stream(torch.npu.current_stream())`   `entry.graph.replay()`   …   `rebind(entry.npu_fia_recorder, entry.npu_update_stream)` |
| **KV 元数据传输与回读裁减** | DiT KV binding、Attention 参数准备 | 打包发布 KV 元数据，并优先使用 Host 已知长度，减少分散 H2D 和设备长度回读。 | `self._storage.copy_(host, non_blocking=event is not None)`   …   `actual_q = actual_seq_lengths_q or _cumulative_lengths(cu_seqlens_q)`   `actual_kv = actual_seq_lengths_kv or _sequence_lengths(seqused_k)` |
| **条件调制缓存与设备驻留** | DiT Block、FinalLayer 条件投影 | 跨 chunk 复用相同评估序号的调制结果；热路径从设备 bank 读取，减少重复投影和参数发布。 | `missing = [i for i, cache in enumerate(caches) if key not in cache]`   …   `caches[index][key] = computed[source : source + 1].clone()`   …   `values = self.modulation_bank.index_select(0, self.modulation_slots.reshape(-1))` |

### 4. Stage 1：Vocoder

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **精确长度解码图与 batch 分桶** | Vocoder Decoder | 按实际 latent 帧数选择解码图，减少大量小算子的重复提交；未覆盖长度回退原生执行。 | `batch, _, frames = latent_ncl.shape`   `batches = self._decode_batches_by_frame.get(int(frames))`   …   `out = slot.run(key, lambda: self.wavegan.audio_encoder(buf))` **不适用于 Decoder，不能作为本项证据；本项保留前两行形状选择代码即可。** |
| **解码输入聚合** | DiT 输出至 Vocoder 的调度 | 累积足够 latent 后再启动解码，减少小增量引起的重复 Decoder 调用；终包允许短尾处理。 | `state.pending = new if state.pending is None else torch.cat((state.pending, new), dim=1)`   `have = 0 if state.pending is None else int(state.pending.shape[1])`   `if have < chunk_frames and not slot.is_last:`   　`continue` |
| 其中第一项可直接采用以下报告代码，避免混入参考 Encoder 路径： |  |  |  |
| `batch, _, frames = latent_ncl.shape` |  |  |  |
| `batches = self._decode_batches_by_frame.get(int(frames))` |  |  |  |
| `if batches is None:` |  |  |  |
| `return self.wavegan.decode_body(latent_ncl)` |  |  |  |

### 5. Host：流水与传输

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **采样结果复用与设备回读裁减** | Stage 0 下一轮输入预处理 | 复用已有 CPU 采样结果，避免额外设备 token 标量回读；结果未就绪时不新增阻塞等待。 | `if not ready.query():`   　`return None`   `token = int(samples[entry[2], 0])`   …   `return {**kwargs, "_omni_current_input_token_id_cpu": token}` |
| **Codec 提前交付** | Stage 0 → Stage 1 | 当前采样被接受后立即构造并交付 Codec 增量，消除等待下一轮 Forward 才发布的流水依赖。 | `codecs.append(token - lo)`   …   `payload["audio_tokens"] = torch.tensor(codecs, dtype=torch.long, device="cpu")` |
| **输出异步搬运与提交线程解耦** | Stage 1 波形输出 | 非阻塞 D2H，并将完成等待移至结果获取阶段，降低输出传输对计算提交线程的阻塞。 | `target.copy_(source, non_blocking=True)`   …   `producer.wait_event(self.event)`   …   `def get_output(self):`   　`self._transfer.wait()` |

## 三、仅在相关测试场景下保留的项目

**参考音频缓存**确实能够减少重复前处理、VQ、Speaker 和参考 VAE 编码，但收益取决于参考复用率。建议仅在报告包含重复参考请求或缓存命中测试时列入，不与无参考、首次请求结果混合。
**主转置卷积相位改写**主要解决 NPU FP32 执行路径问题。没有独立性能对比时，应放入“算子兼容性改造”，不直接称为性能优化成果。
这样分类以**模块范围**为唯一依据，双流、图执行、缓存等作为各模块内部的技术手段。主要清单也不再混入生命周期管理和正确性保护。
