下面补齐为 **“优化项名称｜优化范围｜作用｜关键代码段”**，包含刚核对的 **LLM 图更新路径**，不列文件名。
代码是关键语句摘录，`…` 表示省略；部分条目组合了初始化与运行阶段的语句，**不是可直接运行的完整函数**。

## 一、LLM 与双流执行

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **LLM ACLGraph 捕获** | **Stage 0 / LLM Runner** | 接入 Ascend 框架图捕获，减少模型执行的提交开销。属于框架基础能力。 | `graph_size = NPUModelRunner.capture_model(self)` |
| **LLM 图动态参数更新** | **Stage 0 / LLM Attention 的图执行** | 调用框架 updater，适配动态 Attention 元数据；具体 FIA 分支取决于外部后端和图模式。 | `if update_outer_graph:`   　`with set_current_vllm_config(self.vllm_config):`   　　`self._update_full_graph_params_if_needed(`   　　　`forward_context, num_tokens_padded, positions,`   　　`)` |
| **共享专家双流重叠** | **Stage 0 / LLM MoE** | 共享专家在辅助流计算，允许与路由专家重叠；合并前保留流依赖。 | `with torch.npu.stream(self._stream):`   　`output = self._layer(shared_experts_input)`   `torch.npu.current_stream().wait_stream(self._stream)` |
| **DiT FIA 更新与重放双流协同** | **Stage 1 / Solver、DiT paged Attention** | 图重放和 FIA 参数重绑通过更新流、事件协调，更新真实 KV 长度及页表。 | `entry.npu_update_stream.wait_stream(torch.npu.current_stream())`   `entry.graph.replay()`   …   `rebind(entry.npu_fia_recorder, entry.npu_update_stream)` |
| **跳过无关外层图更新** | **Stage 1 / Runner 与模型自有图衔接** | 模型自主管理求解图时跳过 runner 外层更新，内部 FIA 更新仍保留。 | `update_outer_graph = not self._omni_custom_graph_model or self._omni_custom_graph_needs_runner_update`   …   `if update_outer_graph:` |
| **说明：LLM 的框架更新入口已确认，但不能将其内部实现直接描述为与 DiT 相同的双流方案。** |  |  |  |

## 二、Host 输入、输出与跨 Stage 交付

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **复用 CPU 采样结果** | **Stage 0 / 下一轮输入预处理** | 复用已有采样 D2H 结果，避免额外设备 token 回读；未就绪不主动等待。 | `if not ready.query():`   　`return None`   `token = int(samples[entry[2], 0])`   …   `return {**kwargs, "_omni_current_input_token_id_cpu": token}` |
| **Host 输出索引适配** | **Stage 0 / 输出封装** | 从 CPU 请求边界构造输出索引，不改变模型前向和 sampler 的设备输入。 | `cpu_indices = torch.tensor([end - 1 for end in starts[1:]], dtype=indices.dtype, device="cpu")`   …   `return {**kwargs, "logits_index": cpu_indices}` |
| **平坦 payload 快路** | **Stage 0 / 多模态输出封装** | 直接处理已知字段结构，减少通用拆装；仍保留所有权隔离和必要 D2H。 | `value = values[idx].detach()`   `if value.device.type == "cpu":`   　`payload[key] = value.clone(memory_format=torch.contiguous_format)`   `else:`   　`payload[key] = value.to("cpu").contiguous().clone()` |
| **Codec 提前交付** | **Stage 0 → Stage 1 / Scheduler、输入包构造** | 当前采样被接受后立即发布 Codec，不等待下一轮 LLM Forward。 | `codecs.append(token - lo)`   …   `payload["audio_tokens"] = torch.tensor(codecs, dtype=torch.long, device="cpu")`   `payload["stream_finished"] = torch.tensor([finished or model_finished], dtype=torch.bool, device="cpu")` |
| **首包对齐与条件一次发送** | **Stage 0 → Stage 1 / 输入预处理、发包** | 首包考虑 lookahead 和参考前缀对齐；参考条件不随每个 chunk 重传。 | `return value + lookahead - prefix_tokens % value if first and lookahead else value`   …   `if first_packet:`   　`conditioning = state["conditioning"]`   　`state["conditioning_sent"] = True` |
| **后台收发解耦** | **框架 / 跨 Stage 通信** | 将 connector 收发移出 scheduler 主路径。属于基础异步机制。 | `self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)`   `self.recv_thread.start()`   `self.save_thread = threading.Thread(target=self.save_loop, daemon=True)`   `self.save_thread.start()` |
| **输出异步 D2H、延后等待** | **Stage 1 / 波形等输出交付** | 非阻塞搬运输出，在结果获取阶段等待；通过设备依赖保护源数据。 | `target.copy_(source, non_blocking=True)`   …   `producer.wait_event(self.event)`   …   `def get_output(self):`   　`self._transfer.wait()` |

> 
> Host 输出索引适配与 Talker 已有的 Host 行号推导存在机制重合，不能重复统计两份“去回读”收益。

## 三、Token Encoder 与 DiT KV 输入准备

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **Pinned Token 异步 H2D** | **Stage 1 / Codec Token Encoder 输入** | 将 Python token 数据写入 pinned buffer，再非阻塞搬到 NPU；事件查询复用已完成缓冲区。 | `ready = candidate[2].query()`   …   `host = torch.empty((batch, width), dtype=torch.long, device="cpu", pin_memory=True)`   …   `target_view.copy_(host_view, non_blocking=True)` |
| **就绪增长才编码、等长分组** | **Stage 1 / Codec Token Encoder 调度** | 没有新增安全 token 时不重编码；相同实际 feed 长度的请求组成 batch。 | `if keep > session.encoded_tokens:`   　`groups.setdefault((feed, session.device, session.dtype), []).append((session, keep))`   …   `ids_rows = [session.tokens[:feed] for session, _keep in group]`   `hidden = self._encode_token_rows(ids_rows, device)` |
| **KV 元数据打包发布** | **Stage 1 / DiT paged Attention 输入** | 写入位置、页表等共用固定存储，减少分散 H2D。 | `self.write_idx = self._storage[: batch * q_len]`   `metadata = self._storage[batch * q_len :].view(torch.int32)`   `self.block_table = metadata[: self._table_size].view(rows, nblk_per_slot)`   …   `self._storage.copy_(host, non_blocking=event is not None)` |
| **KV 全局镜像延迟刷新** | **Stage 1 / DiT KV 管理** | snapshot 路径先标 dirty，避免每次更新都刷新全局设备元数据。 | `if getattr(self, "_snapshot_metadata", False):`   　`self._metadata_dirty = True` |
| **KV 形状常量缓存** | **Stage 1 / DiT binding 构造** | 复用固定 batch/query 形状元数据；实际请求 KV 长度仍动态处理。 | `key = (batch, q_len, device)`   `if cache is not None and key in cache:`   　`return cache[key]` |
| **FIA 使用 Host 长度** | **Stage 1 / Attention 参数准备** | 优先使用已知 CPU 长度，避免设备长度 `.tolist()`；缺失时保留回退。 | `actual_q = actual_seq_lengths_q or _cumulative_lengths(cu_seqlens_q)`   `actual_kv = actual_seq_lengths_kv or _sequence_lengths(seqused_k)` |

## 四、图执行、分桶与正确性保护

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **精确 Token Encoder 图** | **Stage 1 / Codec Token Encoder** | 按实际 batch/token 长度重放图，减少编码提交开销；不为命中图 padding 时间维。 | `key = (int(ids.shape[0]), int(ids.shape[1]))`   …   `buffers.copy_(ids)`   …   `result = slot.run(key, lambda: self.encode_tokens(buffers)).clone()` |
| **整窗 ODE 图执行** | **Stage 1 / Solver，包含多次 DiT 前向及更新** | 减少逐步 Host 调度；不是将整个 Stage 1 捕获为一张图。 | `if key is None:`   　`out, histories, raw = self.solve_chunk_body(inputs, pool=pool, method=method, use_cfg=use_cfg)`   `else:`   　`out, histories, raw = self._solve_slot.run(key, lambda: self._solve_body(key))` |
| **多维桶与共享 scratch** | **Stage 1 / Whole-solve Solver 缓冲** | 各 batch/window/history 桶使用最大 scratch 的视图，减少重复分配。 | `batch, width, frames = key`   `inputs = self._solve_scratch.view(batch, width, frames, self.z_dim)`   `static = self._solve_binding(batch, width)`   `inputs.bindings = self._stream_bindings(static, batch, width)` |
| **共享图内存池** | **Stage 1 / Encoder、调制、Solver、Vocoder 等自有图** | 复用图内存池，而不是每类图独立创建 pool；不等于共用一套输入 scratch。 | `if _NPU_GRAPH_POOL is None:`   　`_NPU_GRAPH_POOL = torch.npu.graph_pool_handle()`   `return _NPU_GRAPH_POOL` |
| **图侧输入 prelude** | **Stage 1 / Whole-solve 输入准备** | 在图中清理 padding、设置真实长度、准备历史，减少 Host 逐项提交。 | `valid = (valid_rows[:, None] & valid_frames[None, :]).unsqueeze(-1)`   `inputs.token_hidden.masked_fill_(~valid, 0)`   `inputs.latent_cond.masked_fill_(~valid, 0)`   `inputs.noise.masked_fill_(~valid, 0)`   `inputs.valid_frames.copy_(width)` |
| **阈值 rank 设备化** | **Stage 1 / Solver 输入幅度处理** | 有效计数、分位位置和 gather 留在设备侧，支持变长阈值入图。 | `numerator = (counts.long() - 1) * 19`   `lower = torch.div(numerator, 20, rounding_mode="floor")`   `remainder = numerator.remainder(20)`   `upper = lower + (remainder != 0).long()`   `lo, hi = ordered.gather(1, lower), ordered.gather(1, upper)` |
| **Padding 独立 KV 槽保护** | **Stage 1 / 分桶 Solver、DiT KV** | 为 padding 行保留独立槽位，避免无效行污染真实请求历史。 | `self._solve_pad_slots = [[] for _ in range(streams)]`   …   `for index, group in enumerate(self._solve_pad_slots):`   　`for row in range(max_padding):`   　　`group.append(pool.acquire(f"graph-pad:{index}:{row}"))` |
| **图输出所有权保护** | **Stage 1 / Token、参考 Encoder、调制、Solver、Vocoder** | 持久化前复制图输出，防止后续 replay 覆盖仍在使用的数据。属于正确性保障。 | `result = slot.run(key, lambda: self.encode_tokens(buffers)).clone()`   …   `owned = computed.expand(-1, len(common), -1).clone()` |

## 五、DiT 条件缓存与预计算

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **Time/RTime 频率缓存** | **Stage 1 / DiT 时间条件嵌入** | 缓存正弦嵌入频率，避免每次生成频率向量。 | `self.register_buffer("_freqs", self._make_freqs(), persistent=False)`   …   `args = t.reshape(-1, 1).float() * self._freqs[None]`   `embedding = torch.cat([args.cos(), args.sin()], dim=-1)` |
| **每请求调制缓存** | **Stage 1 / 全部 DiT Block＋FinalLayer 条件投影** | 同 evaluation 的调制量跨 chunk 复用，只计算缺失请求。 | `missing = [i for i, cache in enumerate(caches) if key not in cache]`   `if missing:`   　`computed = self.field_modulations(t, None if speaker is None else speaker[missing], delta_t)`   　…   　`caches[index][key] = computed[source : source + 1].clone()` |
| **冷请求调制图** | **Stage 1 / DiT 调制初始化** | 对首次缺失的调制量使用小图计算，减少首次条件投影的 eager 提交。 | `time_buf, speaker_buf = init.buffers(key)`   `time_buf.copy_(times)`   …   `computed = init.run(key, lambda: self._modulation_body(key))`   `owned = computed.expand(-1, len(common), -1).clone()` |
| **设备调制 bank** | **Stage 1 / Whole-solve → DiT 条件输入** | 热窗口按 slot 读取驻留设备的调制量，减少整份参数重复发布。 | `values = self.modulation_bank.index_select(0, self.modulation_slots.reshape(-1))`   `inputs.modulations.copy_(values.view_as(inputs.modulations))` |
| **ODE 时间网格缓存** | **Stage 1 / Solver 时间输入** | 复用时间节点、步宽及评估时间；与时间嵌入频率缓存不同。 | `cacheable = not torch.is_grad_enabled()`   `if cacheable and key in cache:`   　`return cache[key]`   …   `cache[key] = result` |
| **完整 RoPE 表预计算** | **Stage 1 / Whole-solve DiT 位置输入** | 预先准备覆盖最大 KV 容量的位置表，窗口只读取对应位置。 | `rotary = self.blocks[0].attn.rotary`   `tables = rotary(pool.max_k, scratch.noise.device) if rotary is not None else None` |
| **CFG=0 分支裁减** | **Stage 1 / Token Encoder、DiT Solver** | 关闭 guidance 时跳过额外 empty 编码和无条件评估；下列语句展示条件特征复用。 | `session.token_hidden_empty = empty[:, :frames] if session.solver.cfg_alpha else session.token_hidden` |

## 六、LLM 输出头与参考 Encoder

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **Router bias FP32 缓存** | **Stage 0 / LLM MoE Router** | 缓存已加载 bias 的 FP32 派生值，避免每次路由重复转换。 | `cache = bias.detach().to(torch.float32, copy=True)`   …   `layer.register_buffer(_BIAS_BUFFER, cache, persistent=False)`   …   `kwargs = {**kwargs, "e_score_correction_bias": cache}` |
| **Compact Codec head** | **Stage 0 / LLM 输出头** | 合法候选行组成小 head，避免完整词表投影后丢弃大部分结果。 | `packed = torch.cat(rows, dim=0).contiguous()`   …   `compact.register_buffer("weight", packed, persistent=False)`   …   `compact_logits = talker.logits_processor(self.head, hidden.contiguous())` |
| **Forced token 跳过 head** | **Stage 0 / 输出头、logits 构造** | token 已确定时构造单候选 logits；不是跳过整轮 LLM。 | `if mode is CandidateMode.FORCED and all(item[0] == tid for item in plan):`   　`logits = torch.full((bsz, self.vocab_size), float("-inf"), device=device, dtype=torch.float32)`   　`logits[:, tid].fill_(0.0)`   　`return logits` |
| **同候选模式批处理** | **Stage 0 / 输出头输入整理** | 同模式且无 override 时直接使用整批 hidden，减少逐行选择和拼接。 | `if mode in (CandidateMode.TEXT, CandidateMode.CODEC_ONLY, CandidateMode.CODEC_OR_END):`   　`if all(item[2] is None for item in plan):`   　　`selected_hidden = hidden_states` |
| **候选 mask 缓存** | **Stage 0 / logits 屏蔽** | 复用相同候选范围的 mask，避免每步重新构造。 | `mask = masks.get(key)`   `if mask is None:`   　`mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)`   　…   　`masks[key] = mask` |
| **参考分析缓存** | **参考音频输入预处理** | 命中后复用 mel 和前处理波形，不是 Codec Token Encoder 缓存。 | `entry = cache.get(key)`   `if entry is not None:`   　…   　`return entry["icl_mel"].clone(), entry["padded_reference"].clone()` |
| **参考 VQ 缓存** | **Stage 0 / 参考分析 Encoder＋VQ** | 命中后跳过参考编码、投影和量化，恢复 Codec 索引。 | `entry = cache.get(key)`   `if entry is not None:`   　`return entry["vq"].to(device=self.device, dtype=torch.long)`   …   `cache.put(key, {"vq": vq.detach().cpu()})` |
| **Speaker 条件缓存** | **Stage 1 / Speaker Encoder＋条件映射** | 复用最终 speaker 条件，避免重复参考编码。 | `speaker_states[row].g_cond = entry["g_cond"].to(device=self.token2wav.device)` |
| **Prefix latent 缓存、RNG 恢复** | **Stage 1 / 参考 VAE Encoder** | 跳过重复参考 latent 编码，恢复编码后的 RNG 状态，保持后续随机数位置。 | `prefix_states[row].generator.set_state(entry["generator_state"])`   `targets_by_row[row] = entry["prefix_target"].to(device=self.token2wav.device)` |
| **参考编码等长分组** | **Stage 1 / Speaker、参考 VAE Encoder** | 兼容输入组成 batch，减少逐请求调用；Speaker 按长度和采样率，VAE 按长度分组。 | `for row, (wav, rate) in enumerate(zip(waveforms, sample_rates, strict=True)):`   　`groups.setdefault((wav.numel(), rate), []).append(row)`   …   `groups.setdefault(length, []).append(row)` |
| **参考 VAE 统计图** | **Stage 1 / 参考 VAE Encoder** | 图化确定性 posterior stats，随机采样仍在图外按请求处理。 | `buf.waveform.zero_()`   `buf.length.fill_(key[1] * self.hop_size)`   `return lambda: self._forward_impl(buf.waveform, buf.length)`   …   `registry.register(slot, keys, prepare)` |

## 七、Vocoder 与流式解码

| 优化项名称 | 优化范围 | 作用 | 关键代码段 |
| --- | --- | --- | --- |
| **归一化统计量设备缓存** | **参考 VAE 编码后归一化、FM 输出解码前反归一化** | 同一设备复用 mean/std，避免重复搬运；每次输入的乘加仍执行。 | `cached = (mean.to(device), std.to(device))`   `self._stats_cache[device] = cached` |
| **Decoder context 缓存** | **Stage 1 / Vocoder 窗口准备** | 缓存结构推导的左右上下文，避免重复感受野计算。 | `if self._decoder_context is not None:`   　`return self._decoder_context`   `self._decoder_context = self.wavegan.receptive_field_frames()` |
| **精确时间长度解码图** | **Stage 1 / Vocoder Decoder** | 按真实 latent 帧数查图，不 padding 时间维；未覆盖时回 eager。 | `batch, _, frames = latent_ncl.shape`   `batches = self._decode_batches_by_frame.get(int(frames))`   `if batches is None:`   　`return self.wavegan.decode_body(latent_ncl)` |
| **解码尾包预算** | **Stage 1 / Vocoder batch 分桶** | 限制 EOF 桶的 batch 和 batch-frame 工作量，超限拆 batch。 | `tail_batch = min(self._decode_batch_buckets[-1], tail_max_batch)`   `frame_budget = tail_batch * (left + right + chunk_frames)`   …   `if frames in regular_frames or (b <= tail_batch and (b == 1 or b * frames <= frame_budget))` |
| **Snake 参数预计算** | **Stage 1 / Vocoder Decoder 激活** | 提前物化参数指数和倒数，独立于融合 kernel。 | `self._exp_alpha = torch.exp(self.alpha).contiguous()`   `self._inv_beta = (1.0 / (torch.exp(self.beta) + self.no_div_by_zero)).contiguous()` |
| **FIR 派生权重预计算** | **Stage 1 / Vocoder 抗混叠滤波** | 提前展开通道权重，加载后刷新，避免运行和捕获时重复构造。 | `self.prime_resample_weights()`   `self.register_load_state_dict_post_hook(_refresh_resample_weights)`   …   `self._conv_weight = self.filter.to(dtype=dtype).expand(self.channels, -1, -1).contiguous()` |
| **Weight norm 加载期折叠** | **加载阶段 / 音频 VAE Encoder、Decoder 权重** | 将成对归一化参数物化为普通权重，不仅作用于 Decoder。 | `v = pending_v[stem]`   `folded_weight = torch._weight_norm(v.to(device), g.to(device), 0)`   `out[f"{stem}.weight"] = folded_weight.to(device=v.device, dtype=v.dtype)` |
| **主转置卷积相位改写** | **Stage 1 / Vocoder NPU FP32 主上采样** | 以相位 `Conv1d` 和交织替换转置卷积；不是融合 kernel。 | `phase_taps = weight.shape[-1] // stride`   `weight = weight.reshape(self.in_channels, self.out_channels, phase_taps, stride)`   `weight = weight.permute(1, 3, 0, 2).flip(-1).reshape(-1, self.in_channels, phase_taps)`   `x = F.conv1d(x, weight, padding=phase_taps - 1)`   `x = x.reshape(batch, self.out_channels, stride, -1).transpose(2, 3)` |
| **Latent 累积与解码节拍** | **Stage 1 / DiT 输出 → Vocoder 调度** | 不足节拍先积累，避免小增量反复启动 Decoder；终包允许短尾处理。 | `state.pending = new if state.pending is None else torch.cat((state.pending, new), dim=1)`   `have = 0 if state.pending is None else int(state.pending.shape[1])`   `if have < chunk_frames and not slot.is_last:`   　`continue` |
| **新增波形裁剪、历史裁减** | **Stage 1 / Vocoder 流式输出** | 只发布新增有效波形，裁掉不再需要的历史；必要 context 仍会重算。 | `lo, hi = (emit_lo - win_lo) * hop, (emit_hi - win_lo) * hop`   `parts[row].append(audio[j : j + 1, :, lo : min(hi, audio.shape[-1])])`   …   `keep_from = max(0, emit_hi - left_context_size)`   `if keep_from > 0:`   　`state.history = state.history[:, keep_from:]` |
| **终态资源释放** | **Stage 1 / 请求生命周期、Solver KV** | 完成后移除状态，关闭 FM stream，释放关联槽位。 | `state = self._state_by_req.pop(req_id, None)`   `if state is not None and state.session is not None:`   　`self.token2wav.fm.close_stream(state.session)` |
| **报告口径：上述覆盖主要模块，但属于“实现机制清单”，不是全部已启用或已测得收益的清单。**双流、图桶、compact head 等均有启用条件；仍保留必要事件等待、设备搬运、历史重算和未覆盖形状回退。 |  |  |  |
