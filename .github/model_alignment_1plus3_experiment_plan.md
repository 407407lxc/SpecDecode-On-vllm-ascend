# 模型对齐 1+3 联合实验方案（初学者增强版）

> 版本：v1.1  
> 更新时间：2026-03-10  
> 适用范围：`Qwen3-8B(target) + Qwen3-0.6B(draft)`，NPU 端 V0 `draft_model` spec-decode  
> 约束：本文以“先跑通、再优化”为主，默认不要求你立刻改核心源码

---

## 0. 初学者快速导航（先看这个）

如果你是第一次做这个方向，建议按这个最短路径执行：

1. 先只跑 `Phase 0`，确认脚本和日志链路正常；
2. 只做 `Phase 1` 的 `temperature` 模式（不要一开始就上 affine）；
3. 在 `k=2` 和 `k=4` 各找一个稳定温度后，再进入 `Phase 2/3`。

一句话目标：

- 先学会判断“对齐有没有效果”；
- 再做“对齐 + adaptive-k 联动”。

---

## 1. 背景与目标

本方案对应两个优化方向的联合验证：

1. 方向 1：`logit` 分布对齐（draft 侧校准）  
2. 方向 3：对齐感知的 `adaptive-k` 控制

联合目标：

- 在不显著恶化延迟的前提下，提高 `accept_rate` 与有效吞吐；
- 降低高 `k` 场景下的尾延迟风险和控制器抖动；
- 形成可复现、可比较、可写入毕设正文的实验证据链。

---

## 2. 代码锚点（本方案依赖的现有能力）

### 2.1 方向 1：logit 对齐入口

- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`
  - `_align_logits(...)`
  - 环境变量：
    - `VLLM_ASCEND_DRAFT_ALIGN_ENABLE`
    - `VLLM_ASCEND_DRAFT_ALIGN_MODE`
    - `VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE`
    - `VLLM_ASCEND_DRAFT_ALIGN_SCALE`
    - `VLLM_ASCEND_DRAFT_ALIGN_BIAS`
    - `VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH`

### 2.1.1 给初学者：什么是 logits 对齐

- `logits` 是 softmax 前的原始分数；
- draft 模型分布通常比 target 偏“尖”或偏“平”；
- 通过温度/仿射把 draft 的分布调得更接近 target，可提升被验收概率（`accept_rate`）。

可理解为：

- 你不是改 token 本身，而是在“采样前”改 token 的相对偏好强度。

### 2.1.2 三种模式怎么选（初学者版）

| 模式 | 含义 | 推荐程度 | 适用阶段 |
|---|---|---|---|
| `temperature` | `logits / T` | 高（首选） | 第一轮所有实验 |
| `temperature_then_affine` | 先温度，再 `*scale + bias` | 中 | 温度稳定后做微调 |
| `legacy_affine` | 直接 `*scale + bias` | 低 | 仅做对照 |

重要提醒：

- “全词表同一个常数 bias”通常不会改变 softmax 后概率排序，不能指望靠它带来实质收益。

### 2.1.3 初学者安全参数区间

- `temperature`: `0.90 ~ 1.10`（先从 `0.95/1.00/1.05` 起）
- `scale`: `0.98 ~ 1.02`（只在 `temperature_then_affine` 用）
- `bias`: `0.0`（先固定）

---

### 2.2 方向 3：adaptive-k 入口

- `vllm/vllm/spec_decode/spec_decode_worker.py`
  - `_maybe_update_adaptive_k(...)`
  - `_update_k_ewma(...)`
  - `_select_k_by_utility(...)`
  - 现有可观测日志：
    - `P1P2 k=... step_accept=... waste_ratio=...`
    - `AdaptiveK update: ...`
    - `AdaptiveK utility switch: ...`
    - `ALIGN P1/P2 hs_ratio=...`

### 2.3 基线脚本

- `scripts/model_alignment_bench.sh`
- `scripts/model_alignment_bench_v2.sh`

---

## 3. 关键前提（必须先明确）

对于当前 `Qwen3-8B + Qwen3-0.6B` 组合：

- 存在 `hidden_states` 传递/统计链路；
- 但 `Qwen3` 的 `forward` 不接 `previous_hidden_states` 参数；
- 因此 `hs_available_ratio` 上升，不直接等价于“draft 前向显式使用 hs 条件化”。

这会影响实验解释：

- 方向 1 的主要收益来源应理解为“输出分布对齐”；
- 方向 3 的主要收益来源应理解为“控制策略更稳健”；
- 不应把结果表述为“已完成显式 hidden-state 条件化对齐”。

---

## 4. 研究假设（用于实验判定）

### H1（方向 1）

开启温度类对齐后，`accept_rate` 可提升，且 `iter_latency_ms_p95` 增幅可控。

### H2（方向 3）

在相同负载下，对齐信息与 `adaptive-k` 联动可减少高 `k` 误用，降低尾延迟。

### H3（联合 1+3）

相较单独方向，联合策略在吞吐-时延-接受率三指标上更接近 Pareto 最优。

---

## 5. 指标体系（统一口径）

### 5.1 主指标

- `throughput_tps = accepted_tokens_total / elapsed_seconds`
- `iter_latency_ms_p50/p95 = percentile(spec_step_latency_ms)`
- `accept_rate = accepted_tokens_total / max(proposed_tokens_total, 1)`

### 5.2 对齐指标

- `hidden_states_available_ratio`
- `accept_rate_with_hs`
- `accept_rate_no_hs`
- `delta_accept_hs = accept_rate_with_hs - accept_rate_no_hs`

### 5.3 控制器指标

- `k_histogram`（不同 k 的步数分布）
- `switch_count_per_min`
- `avg_dwell_steps`
- `high_k_occupancy`（例如 k>=4 的占比）

### 5.4 可选补充指标

- `system_efficiency`
- `avg_time_per_proposal_tok_ms / scoring_time_ms / verification_time_ms`

### 5.5 初学者最少要看哪几个

如果你时间有限，至少看这 3 个：

1. `accept_rate`
2. `iter_latency_ms_p95`
3. `throughput_tps`

判定原则：

- 不能只看吞吐，也不能只看接受率；
- 三者至少要满足“两个变好、一个不明显变差”。

---

## 6. 实验分阶段设计

## Phase 0：基线与可观测性校验（必须先跑）

目标：

- 确认日志/指标链路完整；
- 产出 A0/A1/A2 基线可对照结果。

执行：

- 使用 `scripts/model_alignment_bench_v2.sh`
- `FIXED_K_LIST_STR=2,4,8`
- 每个 case 至少 3 次重复

通过条件：

- `summary.csv` 正常落盘；
- server log 中能检索到 `ALIGN P1/P2` 与 `P1P2` 相关行。

---

## Phase 1：方向 1（logit 分布对齐）单独消融

目标：

- 在固定 k 下隔离对齐模块收益；
- 找到稳定的校准参数区间。

### Phase 1-A：新手第一轮（先跑通）

只做这 4 个 case（最小集合）：

1. `k=2, align=off`
2. `k=2, mode=temperature, T=0.95`
3. `k=4, align=off`
4. `k=4, mode=temperature, T=0.95`

这一步的目标不是找最优，而是确认：

- 对齐开关确实生效；
- 你会读结果，不会跑偏。

### Phase 1-B：粗扫（推荐）

建议只用固定 k 先做：

- `k=2`、`k=4`（优先）
- `force_hs=1`（统一为 A1/A2 口径）

实验矩阵（第一轮粗扫）：

- `align_enable`: `0/1`
- `align_mode`: `temperature`
- `temperature`: `0.90/0.95/1.00/1.05`

### Phase 1-C：进阶微调

在温度模式稳定后，再加：

- `align_mode = temperature_then_affine`
- `scale = 0.98/1.00/1.02`
- `bias = 0.0`

第二轮细扫（围绕第一轮最优）：

- `temperature` 以 `0.01~0.02` 步长做局部扫描

### Phase 1-D：结果如何判断（新手判读表）

| 现象 | 可能原因 | 建议动作 |
|---|---|---|
| `accept_rate` 升、`p95` 基本不变 | 对齐有效 | 记录为候选最优 |
| `accept_rate` 升但 `p95` 明显升 | 分布变“过尖/过平”导致尾延迟开销 | 把 T 往 1.0 回调 |
| `accept_rate` 降 | T 方向可能错了 | 若 T<1，尝试增大；若 T>1，尝试减小 |
| 吞吐升但接受率降很多 | 可能在偷吞吐（质量退化） | 不作为最终配置 |
| 指标几乎都不变 | 对齐力度太小或未生效 | 检查环境变量和日志 |

阶段输出：

- 每个 k 的“accept_rate vs temperature”曲线
- 每个 k 的“throughput_tps vs iter_p95”散点图
- 每个 case 的均值/标准差表

---

## Phase 2：方向 3（adaptive-k）单独消融

目标：

- 不开启对齐时，得到 adaptive-k 的稳定基线；
- 识别抖动和高 k 误判场景。

推荐矩阵：

- `VLLM_ASCEND_ADAPTIVE_K_ENABLE=1`
- `VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY=0/1`
- `VLLM_ASCEND_ADAPTIVE_K_INIT=2/4`
- `VLLM_ASCEND_ADAPTIVE_K_MAX=4/8`
- 其余阈值先用当前默认

阶段输出：

- `k_histogram`
- `switch_count_per_min`
- `high_k_occupancy`
- 三主指标对照

---

## Phase 3：联合 1+3（隐式联动，不改控制器公式）

目标：

- 把 Phase 1 最优对齐参数叠加到 Phase 2 最优 adaptive-k 参数；
- 验证“分布对齐”是否能间接改善 adaptive-k 的决策面。

对比组：

1. `B_base`: adaptive-k only（Phase 2 最优）
2. `B_align`: fixed-k + alignment（Phase 1 最优）
3. `B_joint`: adaptive-k + alignment（联合）

重点观察：

- `B_joint` 是否同时改善 `accept_rate` 与 `iter_p95`
- `B_joint` 的 `switch_count_per_min` 是否下降

---

## Phase 4：联合 1+3（显式联动，需最小代码改动，后续可选）

目标：

- 在 `adaptive-k` 决策中显式引入“对齐质量分数”。

建议对齐分数（示例）：

- `align_score = w1 * pos2_accept_ewma + w2 * (1 - waste_ewma) + w3 * delta_accept_hs_ewma`

联动规则（示例）：

- 若 `align_score < low_th`，限制 `k_max=2/3`
- 若 `align_score > high_th` 且队列深度充足，允许上探 `k=4`
- 保持 `min_dwell_steps` 与 `switch_margin` 防抖

说明：

- 该阶段是“扩展实验”，不影响前 3 个阶段先产出论文可用结果。

---

## 7. 实验命令模板（可直接复用）

## 7.0 新手最小跑通命令（建议先跑）

```bash
export RUN_A0=0
export RUN_A1=1
export RUN_A2=1
export FIXED_K_LIST_STR=2,4
export ALIGN_LOG_INTERVAL=200

# 先只用 temperature
export VLLM_ASCEND_DRAFT_ALIGN_MODE=temperature
export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE=0.95
export VLLM_ASCEND_DRAFT_ALIGN_SCALE=1.0
export VLLM_ASCEND_DRAFT_ALIGN_BIAS=0.0

bash scripts/model_alignment_bench_v2.sh
```

## 7.1 方向 1 粗扫（示例）

```bash
export RUN_A0=0
export RUN_A1=1
export RUN_A2=1
export FIXED_K_LIST_STR=2,4
export ALIGN_LOG_INTERVAL=200
export VLLM_ASCEND_SPEC_FORCE_RETURN_HS=1

for T in 0.90 0.95 1.00 1.05; do
  export VLLM_ASCEND_DRAFT_ALIGN_MODE=temperature
  export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE=$T
  export VLLM_ASCEND_DRAFT_ALIGN_SCALE=1.00
  export VLLM_ASCEND_DRAFT_ALIGN_BIAS=0.0
  bash scripts/model_alignment_bench_v2.sh
done
```

## 7.2 方向 1 进阶（加 affine）

```bash
for T in 0.94 0.96 0.98; do
  for S in 0.98 1.00 1.02; do
    export VLLM_ASCEND_DRAFT_ALIGN_MODE=temperature_then_affine
    export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE=$T
    export VLLM_ASCEND_DRAFT_ALIGN_SCALE=$S
    export VLLM_ASCEND_DRAFT_ALIGN_BIAS=0.0
    bash scripts/model_alignment_bench_v2.sh
  done
done
```

## 7.3 方向 3 消融（示例）

```bash
export VLLM_ASCEND_ADAPTIVE_K_ENABLE=1
export VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY=1
export VLLM_ASCEND_ADAPTIVE_K_INIT=4
export VLLM_ASCEND_ADAPTIVE_K_MIN=2
export VLLM_ASCEND_ADAPTIVE_K_MAX=8
export VLLM_ASCEND_ADAPTIVE_MIN_DWELL_STEPS=12
export VLLM_ASCEND_ADAPTIVE_SWITCH_MARGIN=0.03

bash scripts/model_alignment_bench_v2.sh
```

## 7.4 日志抽取建议

```bash
grep -E "ALIGN P1/P2|P1P2|AdaptiveK update|AdaptiveK utility switch|Draft acceptance rate|System efficiency" \
  logs/*/server_*.log > logs/key_lines.txt
```

---

## 8. 结果表模板（建议直接复制到实验报告）

### 8.1 方向 1（固定 k）

| Case | k | mode | T | scale | throughput_tps | iter_p95_ms | accept_rate | delta_accept_hs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |

### 8.2 方向 3（adaptive-k）

| Case | init_k | k_max | utility | throughput_tps | iter_p95_ms | accept_rate | switch/min | high_k_occ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |

### 8.3 联合 1+3

| Case | 配置 | throughput_tps | iter_p95_ms | accept_rate | switch/min | 结论 |
|---|---|---:|---:|---:|---:|---|
| B_base | adaptive only |  |  |  |  |  |
| B_align | align only |  |  |  |  |  |
| B_joint | adaptive + align |  |  |  |  |  |

### 8.4 新手实验记录模板（建议）

| 日期 | case_tag | 主要改动 | 结果摘要 | 下一步 |
|---|---|---|---|---|
|  |  |  |  |  |

---

## 9. 判定门槛（Go/No-Go）

建议门槛分两档：

### 9.1 通过线（初学者可接受）

- `accept_rate` 明显提升，且 `iter_latency_ms_p95` 不恶化超过 `5%`；
- 或 `throughput_tps` 提升 `>=2%` 且 `accept_rate` 不下降。

### 9.2 目标线（论文主结果建议）

1. 方向 1 成立：  
`accept_rate` 提升，且 `iter_latency_ms_p95` 增幅不超过 `3%`。

2. 方向 3 成立：  
`switch_count_per_min` 下降（建议 `>=20%`），且吞吐不下降。

3. 联合 1+3 成立：  
相较 `B_base`，满足以下之一：
- `throughput_tps` 提升（建议 `>=3%`）且 `iter_p95` 不恶化；
- `iter_p95` 下降（建议 `>=8%`）且吞吐不恶化。

---

## 10. 风险与规避

1. 风险：把 `hs_available` 误解为“hs 已被 draft 前向消费”。  
规避：文档与结论中明确区分“可用性”和“消费性”。

2. 风险：参数扫描空间过大，实验周期失控。  
规避：先 `temperature` 单变量，再逐步加维度。

3. 风险：动态负载波动导致结论不稳定。  
规避：每 case 至少 3 次重复，报告均值+标准差。

4. 风险：只看吞吐忽略尾延迟。  
规避：所有结论必须同时给出 `throughput + p95 + accept_rate`。

5. 风险：开关看似生效但实际未生效。  
规避：核对 server log 是否打印 align mode/temperature；核对 `summary.csv` 是否有对齐字段。

---

## 11. 建议执行顺序（最短闭环）

1. 跑 Phase 0（预计半天）
2. 跑 Phase 1-A/1-B（预计 1~2 天）
3. 跑 Phase 2（预计 1 天）
4. 跑 Phase 3（预计 1 天）
5. 若时间允许，再做 Phase 4 扩展

---

## 12. 本文档的预期产出

执行完本方案后，至少应形成以下材料：

1. `summary.csv`（原始指标）  
2. `key_lines.txt`（关键日志）  
3. 三张核心图：
- 对齐参数扫描曲线图
- 吞吐-时延 Pareto 图
- adaptive-k 行为图（k 分布/切换频率）  
4. 一页结论摘要：方向 1、方向 3、联合 1+3 的最终推荐配置。

---

## 附录 A：logits 对齐常见问答（初学者）

### Q1：`T < 1` 和 `T > 1` 分别意味着什么？

- `T < 1`：分布更尖锐，模型更“自信”；
- `T > 1`：分布更平滑，模型更“保守”。

### Q2：为什么我加了 `bias` 看起来没变化？

如果是“给全部词同一个常数”，softmax 后概率相对关系几乎不变，效果很弱。

### Q3：我应该先追求最高吞吐吗？

不建议。对齐方向优先保证 `accept_rate + p95` 稳定，再看吞吐。

### Q4：第一次实验最容易犯什么错？

- 一上来做太大网格；
- 不固定 `k` 就比较对齐参数；
- 只跑 1 次就下结论。

### Q5：我应该从哪个模式开始？

默认从 `temperature` 开始；只有在它稳定提升后，才引入 `temperature_then_affine`。
