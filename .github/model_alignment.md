# NPU 端大小模型对齐（可执行版）

> 文档更新时间：2026-03-08  
> 目标：把“对齐”从概念转成可直接执行、可验收、可回退的实施流程。  
> 约束：本文只定义执行方案，不直接修改源码。

## 1. 目标与范围

“对齐”分两层，按先后执行：

1. Track A（必做）：运行时状态对齐，确保 hidden states / KV 同步链路稳定生效。  
2. Track B（可选）：表示校准，在 draft 输出侧做轻量校准，提高 acceptance。

Go/No-Go 原则：

- 未完成 Track A 验收，不进入 Track B。
- 所有变更必须有开关，且关闭后行为回到 baseline。

---

## 2. 统一指标口径

主指标：

- `throughput_tps = accepted_tokens_total / elapsed_seconds`
- `iter_latency_ms_p50/p95 = percentile(spec_step_latency_ms)`
- `accept_rate = accepted_tokens_total / max(proposed_tokens_total, 1)`

对齐指标：

- `hidden_states_available_ratio = hs_available_spec_steps / total_spec_steps`
- `accept_rate_with_hs = accepted_with_hs / max(proposed_with_hs, 1)`
- `accept_rate_no_hs = accepted_no_hs / max(proposed_no_hs, 1)`

说明：

- `accept_rate_with_hs/no_hs` 使用“token 加权”口径，不使用“step 平均”，避免 batch size 和 proposal 长度差异带来的偏差。

---

## 3. Baseline 实验矩阵（必须先跑）

三组对比：

1. `A0-Default`：默认逻辑，不强制 hidden states。  
2. `A1-ForceHS`：仅开启 `VLLM_ASCEND_SPEC_FORCE_RETURN_HS=1`。  
3. `A2-ForceHS+Calib`：A1 基础上开启 draft 侧校准。

环境变量（PowerShell）：

```powershell
# A0
$env:VLLM_ASCEND_SPEC_FORCE_RETURN_HS="0"
$env:VLLM_ASCEND_DRAFT_ALIGN_ENABLE="0"

# A1
$env:VLLM_ASCEND_SPEC_FORCE_RETURN_HS="1"
$env:VLLM_ASCEND_DRAFT_ALIGN_ENABLE="0"

# A2
$env:VLLM_ASCEND_SPEC_FORCE_RETURN_HS="1"
$env:VLLM_ASCEND_DRAFT_ALIGN_ENABLE="1"
```

每组至少跑 3 次，报告均值与标准差：

| Baseline | throughput_tps | iter_latency_ms_p95 | accept_rate | hidden_states_available_ratio | accept_rate_with_hs | accept_rate_no_hs | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| A0-Default |  |  |  |  |  |  |  |
| A1-ForceHS |  |  |  |  |  |  |  |
| A2-ForceHS+Calib |  |  |  |  |  |  |  |

---

## 4. Problem-Revelation 实验（P1/P2）

### 4.1 P1：hidden states 可用率是否偏低

目标：

- 证明 A0 下 `proposal_scores.hidden_states` 可用率是否明显偏低。
- 同步观察 throughput / latency / acceptance 是否联动变化。

插桩位置：`vllm/vllm/spec_decode/spec_decode_worker.py`

- `SpecDecodeWorker.__init__`：初始化计数器与计时器。
- `_run_speculative_decoding_step`：每步更新统计，周期性打点日志。

关键统计逻辑（推荐口径）：

```python
# 仅统计真实 speculative 行，避免把 non-spec/prefill 混入 proposed 分母
spec_indices = (proposals.proposal_lens > 0).nonzero(as_tuple=False).squeeze(-1)
if spec_indices.numel() > 0:
    proposed = int(proposals.proposal_lens[spec_indices].sum().item())
    accepted = int((accepted_token_ids[spec_indices, :-1] != -1).sum().item())
else:
    proposed, accepted = 0, 0
```

### 4.2 P2：有/无 hidden states 时 acceptance 差异

目标：

- 定量判断 hidden states 链路生效时，acceptance 是否更高。

推荐统计：

```python
has_hs = proposal_scores.hidden_states is not None
if has_hs:
    self.accept_with_hs_proposed += proposed
    self.accept_with_hs_accepted += accepted
else:
    self.accept_no_hs_proposed += proposed
    self.accept_no_hs_accepted += accepted
```

日志每 `N=200` 步输出一次：

- `hs_ratio`
- `accept_rate_with_hs` / `accept_rate_no_hs`
- `throughput_tps`
- `iter_latency_ms_p95`

---

## 5. 现状核对（已确认）

`vllm/vllm/spec_decode/spec_decode_worker.py` 中已有对齐链路：

1. `_run_speculative_decoding_step` 开头把 `self.previous_hidden_states` 传给 proposer。  
2. prefill+decode 混合场景通过 `prepare_prefill_hidden_states(...)` + `prefill_req` 同步 proposer。  
3. `_verify_tokens` 后，若 `proposal_scores.hidden_states` 可用，会更新 `self.previous_hidden_states`。

`vllm-ascend/vllm_ascend/worker/worker.py` 的关键缺口：

- 默认只在 `medusa/mlp_speculator/eagle/deepseek_mtp` 场景启用 `return_hidden_states=True`。
- 通用 draft_model 场景可能长期拿不到 `proposal_scores.hidden_states`。

---

## 6. Track A 实施清单（运行时状态对齐）

涉及文件：

- `vllm-ascend/vllm_ascend/worker/worker.py`
- `vllm/vllm/spec_decode/spec_decode_worker.py`

### 6.1 变更项 A：强制返回 hidden states 的开关

建议开关：`VLLM_ASCEND_SPEC_FORCE_RETURN_HS=1`

实现要求：

- 仅在 `speculative_config is not None` 时生效。
- 以“补充字段”方式设置，不覆盖其他 `speculative_args`。
- 默认关闭，避免影响现有路径。

### 6.2 变更项 B：增加运行时可观测性

新增计数器：

- `total_spec_steps`
- `hidden_states_available_steps`
- `accepted_tokens_total` / `proposed_tokens_total`
- `accept_with_hs_accepted/proposed`
- `accept_no_hs_accepted/proposed`
- `iter_latency_ms[]`

日志策略：

- 仅 rank0 周期打印。
- 与现有统计频率对齐，避免日志风暴。

### 6.3 变更项 C：与动态 k 联动（可选）

当满足以下条件时，优先降 `k`：

- `hidden_states` 连续不可用达到阈值（例如 32 步）。
- 且 `accept_rate` 低于近期 EWMA 阈值。

注意：

- 只调整 runtime cap，不改 scheduler 的全局配置。

### 6.4 Track A 验收门槛

1. A1 对比 A0：`hidden_states_available_ratio` 明显提升（建议绝对值 +20% 以上）。  
2. A1 对比 A0：`accept_rate` 不下降，优先要求提升。  
3. A1 对比 A0：`throughput_tps` 提升，或 `iter_latency_ms_p95` 下降。  
4. prefill/decode 混合 batch 无新增崩溃路径。

---

## 7. Track B 实施清单（表示校准，可选）

涉及文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

插入点：`TP1DraftModelRunner.execute_model` 中 `compute_logits(...)` 之后、`sampler(...)` 之前。

### 7.1 M0/M1/M2 方案

1. M0（首选起步）：温度校准 `logits / T`（`T > 0`）。  
2. M1（推荐）：hidden 通道仿射（小参数量）。  
3. M2（扩展）：低秩投影校准。

重要修正：

- 不建议使用“全词表同一标量 bias”（`logits + b`）；该操作对 softmax 分布不产生实质变化。

### 7.2 Track B 验收门槛

1. A2 对比 A1：`accept_rate` 提升。  
2. A2 对比 A1：`throughput_tps` 无显著下降。  
3. A2 对比 A1：`iter_latency_ms_p95` 增量可控（建议 < 3%）。  
4. 关闭开关后可完全回退到 A1。

---

## 8. 执行命令与产出物

基础测试：

```powershell
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_spec_decode_worker.py -q
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_multi_step_worker.py -q
```

产出物（每轮实验必须提交）：

1. A0/A1/A2 指标表（均值、标准差、样本数）。  
2. 关键日志片段（P1/P2 周期日志）。  
3. 结论：是否进入下一 Track（Go/No-Go）。

---

## 9. 风险与规避

1. 统计口径风险：`proposed` 分母混入 non-spec，会误判 acceptance。  
2. 稳定性风险：零 proposal step 时，依赖 `step_accept` 的逻辑需防空。  
3. 性能风险：开启 hidden states 会增加回传与内存压力，必须以 A0/A1 实测为准。  
4. 回归风险：所有新逻辑必须受环境变量控制，默认关闭。

---

## 10. 不建议做法

1. 引入仓库不存在的“通用清理 API”并强依赖。  
2. 每步无条件额外前向同步 proposer（吞吐风险高）。  
3. 未完成 P1/P2 与三主指标联动验证就宣称“对齐完成”。

---

## 11. 校准模式开关（新增）

`vllm-ascend/vllm_ascend/worker/draft_model_runner.py` 已支持以下校准参数：

- `VLLM_ASCEND_DRAFT_ALIGN_MODE`: `legacy_affine` / `temperature` / `temperature_then_affine`
- `VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE`: 温度系数（>0）
- `VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH`: 可选词表 bias 向量路径（`.pt`）

推荐优先从 `temperature` 模式开始：

```bash
export VLLM_ASCEND_DRAFT_ALIGN_ENABLE=1
export VLLM_ASCEND_DRAFT_ALIGN_MODE=temperature
export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE=0.95
```

你可以直接用 `scripts/model_alignment_bench_v2.sh` 复用现有 A0/A1/A2 流程：

```bash
bash scripts/model_alignment_bench_v2.sh
```

---

## 12. HS 真对齐定义与可行性（合并自 hs_alignment_prompt）

### 12.1 术语边界（必须统一）

- `hs_available`: target 返回了 hidden states，且链路已传递到 draft worker。  
- `hs_effective`: draft 在本步前向中真实消费了 `previous_hidden_states` 并参与 token 决策。  

结论：

- 只提升 `hs_available_ratio` 不等于完成 HS 对齐。  
- 实验结论必须同时报告 `hs_available_ratio` 与 `hs_effective_ratio`。

### 12.2 当前代码状态（已核对）

- HS 传递链路已具备：`spec_decode_worker -> ExecuteModelRequest -> draft runner`。  
- `Qwen3` 当前 `forward` 不接 `previous_hidden_states`，默认不会消费 HS。  
- `deepseek_mtp` 是已消费 HS 的可参考实现。

技术可行性判断：**Yes**。  
理由：基础链路与能力探测均已存在，仅缺“Qwen3 侧消费逻辑（或 runner 融合逻辑）”。

---

## 13. 两条落地路径（先快后稳）

### 13.1 路径 A：Runner 融合（快速 POC）

目标：不改 Qwen3 模型签名，在 `TP1DraftModelRunner` 内融合 HS。  

推荐张量流：

1. 取 `h_prev = previous_hidden_states`，形状 `[B, d_t]`。  
2. 取 `h_cur = hidden_states`，形状 `[B, d_d]`。  
3. `h_proj = W_h(LN(h_prev))`，将 `d_t -> d_d`。  
4. `h_fused = h_cur + sigmoid(g) * h_proj`。  
5. `compute_logits(h_fused)` 再采样。

推荐开关（默认全关）：

- `VLLM_ASCEND_DRAFT_HS_ENABLE=0/1`
- `VLLM_ASCEND_DRAFT_HS_ADAPTER_PATH=/path/to/adapter.pt`
- `VLLM_ASCEND_DRAFT_HS_GATE_INIT=0.0`
- `VLLM_ASCEND_DRAFT_HS_FUSE_MODE=add_gate`（预留扩展）

优点：改动最小、最快拿到正反证据。  
风险：语义位于 runner 层，长期维护性一般。

### 13.2 路径 B：Model 融合（正式版）

目标：在 `CustomQwen3Model/ForCausalLM.forward` 显式支持  
`previous_hidden_states: Optional[torch.Tensor] = None`，在模型内部融合。

推荐张量流：

1. `inputs_embeds = embed_tokens(input_ids)`  
2. 若 `previous_hidden_states is not None`：  
`inputs_embeds += sigmoid(g) * hs_adapter(LN(previous_hidden_states))`
3. 进入 decoder layers。

优点：语义清晰，论文表述完整。  
风险：改动范围更大，需更多回归测试。

---

## 14. 最小改动清单（文件/函数）

### 14.1 路径 A（最小改动）

- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`
  - `TP1DraftModelRunner.__init__`: 新增 HS 融合开关、adapter 加载、gate 参数。
  - `TP1DraftModelRunner.execute_model`: 在 `compute_logits(...)` 前融合 HS。
  - 新增私有函数：`_fuse_hs_into_hidden(...)`、`_log_hs_effective_metrics(...)`。

- `scripts/model_alignment_bench.sh`
  - 解析并落盘 `hs_effective_ratio`（新增列）。
  - 保留现有 `ALIGN P1/P2` 统计字段。

### 14.2 路径 B（正式改动）

- `vllm-ascend/vllm_ascend/models/qwen3.py`
  - `CustomQwen3Model.__init__`: 新增 `hs_adapter` / gate。
  - `CustomQwen3Model.forward`: 增加 `previous_hidden_states` 可选输入并融合。
  - `CustomQwen3ForCausalLM.forward`: 透传 `previous_hidden_states`。

- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`
  - 继续复用已有 `_supports_previous_hidden_states` 探测逻辑。

---

## 15. 训练方案（仅训小适配层）

目标：避免重训 0.6B 主干，只训练可热插拔 adapter。  

训练设置建议：

1. 冻结 draft 主干参数。  
2. 仅训练 `hs_adapter` 与 gate。  
3. 损失函数：  
   - `KL(student_logits || teacher_logits)`  
   - `CE(student_logits, gt)`  
   - 可选 `MSE(h_adapter, h_target_proj)`  
4. 产物：`adapter.pt`（独立权重，可开关加载）。

训练数据建议：

- 优先复用你现有 benchmark 请求分布（ShareGPT 同口径）。  
- 记录 `(input_ids, previous_hidden_states, teacher_logits, gt)`。  
- 训练/验证按请求切分，避免同请求泄漏。

---

## 16. HS 专项实验矩阵与显著性门槛

### 16.1 实验组（每组至少 3 次重复）

1. `H0`: ForceHS only（有 HS 传递，无消费）。  
2. `H1`: 路径 A，无训练 adapter。  
3. `H2`: 路径 A + 训练 adapter。  
4. `H3`: 路径 B，无训练 adapter。  
5. `H4`: 路径 B + 训练 adapter。  

每组扫 `k=2/4/8`，并统一记录：

- `throughput_tps`
- `iter_latency_ms_p95`
- `accept_rate`
- `hidden_states_available_ratio`
- `hs_effective_ratio`
- `accept_rate_with_hs / accept_rate_no_hs`

### 16.2 基于现有数据的效果预测（Qwen3-8B + Qwen3-0.6B）

从 `github/Data` 现有结果可见：`k=2` 已较高，`k=4/8` 仍有明显优化空间。  

预测区间（经验 + 现有曲线）：

- `k=2`: `accept_rate` 提升约 `0~1.5pp`，吞吐提升约 `-1%~+3%`。  
- `k=4`: `accept_rate` 提升约 `1.5~3.5pp`，吞吐提升约 `+5%~+12%`。  
- `k=8`: `accept_rate` 提升约 `2~5pp`，吞吐提升约 `+3%~+10%`（方差较大）。

### 16.3 显著性判定（建议）

- 主判定（推荐用于正文）：  
  - `accept_rate` 绝对提升 `>= 2pp`，且  
  - `throughput_tps` 提升 `>= 5%`，且  
  - `iter_latency_ms_p95` 恶化不超过 `3%`。  

- 统计判定：  
  - 同配置重复 `n>=3`，报告均值±标准差；  
  - 若可行，补充 bootstrap 95% CI（CI 不跨 0 视为显著）。

---

## 17. 回归测试点与回退策略

回归测试点：

1. `spec_decode_worker` 基础功能测试通过。  
2. `multi_step_worker` 在 bonus token 与 prefill/decode 混合场景不回退。  
3. 开关关闭后，行为与现 baseline 一致（bitwise 不强求，指标一致即可）。

回退策略：

- 任一异常（shape 不匹配、adapter 载入失败、性能退化超阈值），  
  立即置 `VLLM_ASCEND_DRAFT_HS_ENABLE=0` 回到无 HS 消费路径。  
- 保持新逻辑“可拔插、可禁用、可单点回滚”。

---

## 18. adaptive-k 全量数据整合（来自 `github/Data/adaptive-k`）

### 18.1 数据完整性与口径检查

- 报告文件数：11（fixed/adaptive/adaptive_pro 全部读取）
- Successful requests 全部为 200：是
- Total input tokens 全部为 43560：是
- Total generated tokens 观测到的取值：44524, 44636, 44697
- 说明：spec0 是 no-spec 基线，输出 token 与 spec 组不必严格一致；spec8-fixed 记录为 44636，其余 spec 组多为 44697。

### 18.2 全量主指标总表（11 组）

| Case | mode | k | Duration(s) | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Mean ITL | P99 ITL | Draft acceptance | System efficiency | Adaptive updates | Tail final k |
| ---- | ---- | -: | ----------: | -----------: | ----------: | --------: | --------: | -------: | ------: | ---------------: | ----------------: | ---------------: | -----------: |
| spec0-fixed | fixed | 0 | 63.21 | 704.33 | 1393.41 | 237.24 | 149.94 | 94.63 | 492.31 | - | - | 0 | - |
| spec2-fixed | fixed | 2 | 107.77 | 414.74 | 818.92 | 254.25 | 253.54 | 357.89 | 1549.36 | 0.776 | 0.804 | 0 | 2 |
| spec2-adaptive | adaptive | 2 | 118.57 | 376.97 | 744.35 | 270.51 | 304.55 | 451.97 | 2095.75 | 0.762 | 0.681 | 8 | 2 |
| spec2-adaptive_pro | adaptive_pro | 2 | 117.22 | 381.31 | 752.91 | 296.92 | 303.35 | 414.15 | 2637.54 | 0.776 | 0.744 | 11 | 2 |
| spec4-fixed | fixed | 4 | 138.66 | 322.34 | 636.48 | 321.05 | 298.63 | 575.14 | 2777.87 | 0.747 | 0.648 | 0 | 4 |
| spec4-adaptive | adaptive | 4 | 117.86 | 379.23 | 748.82 | 278.48 | 273.95 | 447.46 | 2075.64 | 0.762 | 0.673 | 8 | 2 |
| spec4-adaptive_pro | adaptive_pro | 4 | 114.95 | 388.84 | 767.78 | 291.64 | 238.65 | 392.81 | 1945.47 | 0.765 | 0.720 | 11 | 2 |
| spec8-fixed | fixed | 8 | 192.27 | 232.16 | 458.71 | 567.15 | 479.52 | 979.48 | 5562.85 | 0.717 | 0.446 | 0 | 8 |
| spec8-adaptive | adaptive | 8 | 117.86 | 379.22 | 748.80 | 243.64 | 249.91 | 442.43 | 1788.54 | 0.763 | 0.672 | 12 | 2 |
| spec8-adaptive_pro | adaptive_pro | 8 | 121.75 | 367.12 | 724.89 | 449.82 | 372.54 | 521.19 | 4398.90 | 0.739 | 0.579 | 11 | 2 |
| spec5-fixed | fixed | 5 | 153.46 | 291.26 | 575.12 | 539.43 | 300.49 | 640.03 | 2450.38 | 0.737 | 0.588 | 0 | - |

### 18.3 同一 k 的胜负结论（fixed vs adaptive vs adaptive_pro）

| k | 吞吐最优(Output tok/s) | TTFT 最优 | TPOT 最优 | ITL 最优 | 质量最优(accept/eff) | 结论 |
| -: | --------------------- | --------- | --------- | -------- | -------------------- | ---- |
| 2 | fixed (414.74) | fixed (254.25) | fixed (253.54) | fixed (357.89) | fixed (0.776/0.804) | k=2 下 fixed 仍是性能上限，adaptive_pro 仅在 eff 上更接近最优。 |
| 4 | adaptive_pro (388.84) | adaptive (278.48) | adaptive_pro (238.65) | adaptive_pro (392.81) | adaptive_pro (0.765/0.720) | k=4 下 adaptive_pro 吞吐与 TPOT/ITL 最优，是当前最推荐工作点。 |
| 8 | adaptive (379.22) | adaptive (243.64) | adaptive (249.91) | adaptive (442.43) | adaptive (0.763/0.672) | k=8 下 adaptive 明显优于 fixed；adaptive_pro 出现退化。 |

### 18.4 三类增益百分比（核心对比）

A) `adaptive` 相对 `fixed`（同 k）

| k | Output tok/s Δ | Mean TTFT Δ | Mean TPOT Δ | Mean ITL Δ | P99 ITL Δ | acceptance Δ(pp) | efficiency Δ(pp) |
| -: | -------------: | ----------: | ----------: | ---------: | --------: | ----------------: | ----------------: |
| 2 | -9.11% | 6.40% | 20.12% | 26.29% | 35.27% | -1.40 | -12.30 |
| 4 | 17.65% | -13.26% | -8.26% | -22.20% | -25.28% | 1.50 | 2.50 |
| 8 | 63.34% | -57.04% | -47.88% | -54.83% | -67.85% | 4.60 | 22.60 |

B) `adaptive_pro` 相对 `adaptive`（同 k）

| k | Output tok/s Δ | Mean TTFT Δ | Mean TPOT Δ | Mean ITL Δ | P99 ITL Δ | acceptance Δ(pp) | efficiency Δ(pp) |
| -: | -------------: | ----------: | ----------: | ---------: | --------: | ----------------: | ----------------: |
| 2 | 1.15% | 9.76% | -0.39% | -8.37% | 25.85% | 1.40 | 6.30 |
| 4 | 2.53% | 4.73% | -12.89% | -12.21% | -6.27% | 0.30 | 4.70 |
| 8 | -3.19% | 84.62% | 49.07% | 17.80% | 145.95% | -2.40 | -9.30 |

C) `adaptive_pro` 相对 `fixed`（同 k）

| k | Output tok/s Δ | Mean TTFT Δ | Mean TPOT Δ | Mean ITL Δ | P99 ITL Δ | acceptance Δ(pp) | efficiency Δ(pp) |
| -: | -------------: | ----------: | ----------: | ---------: | --------: | ----------------: | ----------------: |
| 2 | -8.06% | 16.78% | 19.65% | 15.72% | 70.23% | 0.00 | -6.00 |
| 4 | 20.63% | -9.16% | -20.09% | -31.70% | -29.97% | 1.80 | 7.20 |
| 8 | 58.13% | -20.69% | -22.31% | -46.79% | -20.92% | 2.20 | 13.30 |

### 18.5 相对 spec0 基线差值（用于论文总对照）

| Case | Output tok/s Δ | Mean TTFT Δ | Mean TPOT Δ | Mean ITL Δ |
| ---- | -------------: | ----------: | ----------: | ---------: |
| spec2-adaptive | -46.48% | 14.02% | 103.11% | 377.62% |
| spec2-adaptive_pro | -45.86% | 25.16% | 102.31% | 337.65% |
| spec2-fixed | -41.12% | 7.17% | 69.09% | 278.20% |
| spec4-adaptive | -46.16% | 17.38% | 82.71% | 372.85% |
| spec4-adaptive_pro | -44.79% | 22.93% | 59.16% | 315.10% |
| spec4-fixed | -54.23% | 35.33% | 99.17% | 507.78% |
| spec5-fixed | -58.65% | 127.38% | 100.41% | 576.35% |
| spec8-adaptive | -46.16% | 2.70% | 66.67% | 367.54% |
| spec8-adaptive_pro | -47.88% | 89.61% | 148.46% | 450.77% |
| spec8-fixed | -67.04% | 139.06% | 219.81% | 935.06% |

### 18.6 动态行为统计（adaptive 与 adaptive_pro）

| Case | mode | base k | Adaptive updates | Tail final k | Tail final waste_ratio | Tail step_accept |
| ---- | ---- | -----: | ---------------: | -----------: | ---------------------: | ---------------- |
| spec2-adaptive | adaptive | 2 | 8 | 2 | 0.1291 | [1.0, 0.7418] |
| spec2-adaptive_pro | adaptive_pro | 2 | 11 | 2 | 0.1161 | [1.0, 0.7679] |
| spec4-adaptive | adaptive | 4 | 8 | 2 | 0.1380 | [1.0, 0.7239] |
| spec4-adaptive_pro | adaptive_pro | 4 | 11 | 2 | 0.1351 | [1.0, 0.7298] |
| spec8-adaptive | adaptive | 8 | 12 | 2 | 0.1565 | [1.0, 0.6869] |
| spec8-adaptive_pro | adaptive_pro | 8 | 11 | 2 | 0.1314 | [1.0, 0.7373] |

- 观测到的共性：6 组动态实验在尾段全部回落到 `k=2`。
- 风险提示：`spec8-adaptive_pro` 出现明显退化（TTFT/TPOT/ITL 均恶化），应在策略里限制高 k 触发条件。

结论：
1. 在当前数据下，`k=4 + adaptive_pro` 是 speculative 组综合最优点（吞吐最高，TPOT/ITL 较优）。
2. `k=2` 区间 fixed 仍有优势，说明动态控制在低 k 负载下还有开销。
3. `k=8` 区间 adaptive_pro 不稳定，建议先将 `k_max` 收敛到 `4` 再迭代高 k 门控。
