# NPU 端 draft_model 动态步长优化（实测驱动 + 全量横向对比）

> 文档更新时间：2026-03-08  
> 代码基线：`vllm/vllm/spec_decode/spec_decode_worker.py`（AdaptiveK + adaptive_pro）  
> 数据来源：`github/Data/*.md`

---

## 0. 快速结论

1. 在你当前环境和模型组合下，`adaptive_pro` 的最优点是 `base_k=4`。  
2. `adaptive_pro` 不是全局优于 `adaptive`：
   - `k=4` 时明显更好（吞吐更高，TPOT/ITL 更低）。
   - `k=8` 时明显更差（TTFT/TPOT/ITL 大幅恶化）。
3. 所有 speculative 方案仍显著落后 `spec0`（no-spec）基线，说明控制器和 draft/scorer 成本还需继续压。

---

## 1. 数据与指标口径

数据目录：`github/Data/`（共 11 份报告）

- `fixed`：`spec0/spec2/spec4/spec5/spec8`
- `adaptive`：`spec2/spec4/spec8`
- `adaptive_pro`：`spec2/spec4/spec8`

统一指标：

- 吞吐：`Request throughput`, `Output token throughput`, `Total token throughput`
- 时延：`Mean/P99 TTFT`, `Mean/P99 TPOT`, `Mean/P99 ITL`
- spec 质量：`Draft acceptance rate`, `System efficiency`（取末尾稳定值）

---

## 2. 全量结果总表（全面横向）

### 2.1 吞吐总表

| Case | mode | base k | Duration(s) | Req/s | Output tok/s | Total tok/s |
|---|---|---:|---:|---:|---:|---:|
| spec0 | fixed | 0 | 63.21 | 3.16 | 704.33 | 1393.41 |
| spec2 | fixed | 2 | 107.77 | 1.86 | 414.74 | 818.92 |
| spec2 | adaptive | 2 | 118.57 | 1.69 | 376.97 | 744.35 |
| spec2 | adaptive_pro | 2 | 117.22 | 1.71 | 381.31 | 752.91 |
| spec4 | fixed | 4 | 138.66 | 1.44 | 322.34 | 636.48 |
| spec4 | adaptive | 4 | 117.86 | 1.70 | 379.23 | 748.82 |
| spec4 | adaptive_pro | 4 | 114.95 | 1.74 | 388.84 | 767.78 |
| spec5 | fixed | 5 | 153.46 | 1.30 | 291.26 | 575.12 |
| spec8 | fixed | 8 | 192.27 | 1.04 | 232.16 | 458.71 |
| spec8 | adaptive | 8 | 117.86 | 1.70 | 379.22 | 748.80 |
| spec8 | adaptive_pro | 8 | 121.75 | 1.64 | 367.12 | 724.89 |

### 2.2 时延总表（Mean + P99）

| Case | mode | k | Mean TTFT | Mean TPOT | Mean ITL | P99 TTFT | P99 TPOT | P99 ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| spec0 | fixed | 0 | 237.24 | 149.94 | 94.63 | 511.24 | 329.87 | 492.31 |
| spec2 | fixed | 2 | 254.25 | 253.54 | 357.89 | 559.71 | 789.67 | 1549.36 |
| spec2 | adaptive | 2 | 270.51 | 304.55 | 451.97 | 549.86 | 1377.33 | 2095.75 |
| spec2 | adaptive_pro | 2 | 296.92 | 303.35 | 414.15 | 732.43 | 1396.90 | 2637.54 |
| spec4 | fixed | 4 | 321.05 | 298.63 | 575.14 | 1166.47 | 912.29 | 2777.87 |
| spec4 | adaptive | 4 | 278.48 | 273.95 | 447.46 | 608.21 | 827.70 | 2075.64 |
| spec4 | adaptive_pro | 4 | 291.64 | 238.65 | 392.81 | 563.34 | 657.98 | 1945.47 |
| spec5 | fixed | 5 | 539.43 | 300.49 | 640.03 | 1815.78 | 859.68 | 2450.38 |
| spec8 | fixed | 8 | 567.15 | 479.52 | 979.48 | 1967.95 | 2021.05 | 5562.85 |
| spec8 | adaptive | 8 | 243.64 | 249.91 | 442.43 | 461.93 | 657.14 | 1788.54 |
| spec8 | adaptive_pro | 8 | 449.82 | 372.54 | 521.19 | 1760.13 | 1591.21 | 4398.90 |

### 2.3 Spec 质量指标（末尾稳定值）

| Case | mode | k | Draft acceptance | System efficiency |
|---|---|---:|---:|---:|
| spec2 | adaptive | 2 | 0.762 | 0.681 |
| spec2 | adaptive_pro | 2 | 0.776 | 0.744 |
| spec4 | adaptive | 4 | 0.762 | 0.673 |
| spec4 | adaptive_pro | 4 | 0.765 | 0.720 |
| spec5 | fixed | 5 | 0.737 | 0.588 |
| spec8 | adaptive | 8 | 0.763 | 0.672 |
| spec8 | adaptive_pro | 8 | 0.739 | 0.579 |

---

## 3. 同一 k 下的全面对比（fixed vs adaptive vs adaptive_pro）

### 3.1 胜负表（同 k）

| k | 吞吐最优（Output/Total） | 时延最优（TTFT/TPOT/ITL） | 质量最优（Acc/Eff） | 综合结论 |
|---:|---|---|---|---|
| 2 | fixed | fixed | adaptive_pro | `k=2` 下 fixed 性能最强，adaptive_pro 仅质量领先 |
| 4 | adaptive_pro | TTFT: adaptive；TPOT/ITL: adaptive_pro | adaptive_pro | `k=4` 下 adaptive_pro 综合最好 |
| 8 | adaptive | adaptive | adaptive | `k=8` 下 adaptive_pro 退化明显 |

### 3.2 `adaptive_pro` 对 `adaptive`（同 base k）增益百分比

> 说明：吞吐 `+` 更好；时延 `%` 为负代表更低（更好），为正代表更高（更差）。

| k | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Mean ITL | P99 TTFT | P99 TPOT | P99 ITL | Acc Δ | Eff Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +1.15% | +1.15% | +9.76% | -0.39% | -8.37% | +33.20% | +1.42% | +25.85% | +0.014 | +0.063 |
| 4 | +2.53% | +2.53% | +4.73% | -12.89% | -12.21% | -7.38% | -20.51% | -6.27% | +0.003 | +0.047 |
| 8 | -3.19% | -3.19% | +84.62% | +49.07% | +17.80% | +281.04% | +142.14% | +145.95% | -0.024 | -0.093 |

### 3.3 `adaptive_pro` 对 `fixed`（同 k）增益百分比

| k | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Mean ITL |
|---:|---:|---:|---:|---:|---:|
| 2 | -8.06% | -8.06% | +16.78% | +19.65% | +15.72% |
| 4 | +20.63% | +20.63% | -9.16% | -20.09% | -31.70% |
| 8 | +58.13% | +58.03% | -20.69% | -22.31% | -46.79% |

---

## 4. `adaptive_pro` 内部横向（2 vs 4 vs 8）

| adaptive_pro | k=2 | k=4 | k=8 | 最优 |
|---|---:|---:|---:|---|
| Output tok/s | 381.31 | **388.84** | 367.12 | k=4 |
| Total tok/s | 752.91 | **767.78** | 724.89 | k=4 |
| Mean TTFT | 296.92 | **291.64** | 449.82 | k=4 |
| Mean TPOT | 303.35 | **238.65** | 372.54 | k=4 |
| Mean ITL | 414.15 | **392.81** | 521.19 | k=4 |
| Draft acceptance | **0.776** | 0.765 | 0.739 | k=2 |
| System efficiency | **0.744** | 0.720 | 0.579 | k=2 |

解释：

- `k=4` 的吞吐和延迟最平衡，是当前最优工作点。
- `k=2` 的质量指标最高，但吞吐略低于 `k=4`。
- `k=8` 出现“高 k 失稳”迹象（效率下降、尾时延恶化）。

---

## 5. 与 no-spec（spec0）基线对比

| Case | Output tok/s | Mean TTFT | Mean TPOT | Mean ITL |
|---|---:|---:|---:|---:|
| spec0 | 704.33 | 237.24 | 149.94 | 94.63 |
| best spec（adaptive_pro k=4） | 388.84 | 291.64 | 238.65 | 392.81 |
| 差距（best spec 相对 spec0） | -44.79% | +22.94% | +59.16% | +315.09% |

结论：当前阶段的核心矛盾不是“选 2/4/8”，而是 speculative pipeline 的总成本偏高，尤其是 ITL。

---

## 6. 进一步优化思路（代码层面可执行）

## 6.1 方向 A：把目标函数改成“吞吐-时延联合效用”而不是单阈值

当前问题：`k=8` 在 acceptance 看起来不低时，仍可能导致 TPOT/ITL 爆炸。  
改法：在 `_maybe_update_adaptive_k` 里显式加入时延惩罚项：

```python
# utility 越大越好
# emit_per_ms: 期望每毫秒产生的token
# tail_penalty: P99惩罚项，防止高k尾延迟爆炸
utility = emit_per_ms - alpha * mean_tpot_ms - beta * mean_itl_ms - gamma * p99_tpot_ms
```

建议初始权重：`alpha=0.002, beta=0.0015, gamma=0.0008`，再用网格搜索调。

## 6.2 方向 B：限制高 k 的进入条件（k=8 保护门）

从数据看，`k=8` 只有在很强条件下才值得尝试。建议新增硬门限：

```python
allow_k8 = (
    queue_depth >= 96 and
    pos2_accept_ewma >= 0.82 and
    pos3_accept_ewma >= 0.72 and
    waste_ewma <= 0.18 and
    p99_tpot_ms <= p99_tpot_budget_ms
)
if not allow_k8:
    k = min(k, 4)
```

## 6.3 方向 C：切换抗抖动（减少 2<->3/4 来回震荡）

增加双向滞回与最小驻留步数：

```python
if step_id - last_switch_step < min_dwell_steps:
    return current_k

# 升档需要更严格条件，降档更快触发
up_margin = 0.06
down_margin = 0.02
```

目标：每分钟切换次数下降至少 30%。

## 6.4 方向 D：分阶段策略（预热/稳态/尾段）

按队列深度分 3 段，固定候选集合，减少策略复杂度：

- 高负载（`q>=96`）：候选 `{3,4}`，默认 4
- 中负载（`32<=q<96`）：候选 `{2,3,4}`，默认 3
- 尾段（`q<32`）：固定 `k=2`

这能直接规避你日志里尾段高 k 拖慢的问题。

## 6.5 方向 E：实验设计升级（确保结论稳定）

下一轮矩阵建议：

- `request_rate`: `16 / 24 / 32`
- 每组 `3` 次重复
- 固定对照：`fixed k=2`, `fixed k=4`
- 动态组：`adaptive_pro init=2/4`, `k_max=4/5/8`

输出必须包含：

- 全局：`Output tok/s`, `Mean/P99 TTFT/TPOT/ITL`, `Acc/Eff`
- 稳态窗口（只取中段 `running_reqs>=64`）：同样一组指标
- 控制器行为：`switch_count/min`, `k_histogram`

---

## 7. 下一步执行优先级

1. 先把默认策略收敛到 `init=4, k_max=4`（先拿稳定收益）。  
2. 在此基础上加 `k=8` 保护门和联合效用函数。  
3. 跑完整矩阵后再决定是否保留 `k_max=8`。  
4. 若仍显著落后 `spec0`，转入 `model_alignment`（降低 draft/verify mismatch 成本）。  

---

## 8. 改动留档（Adaptive vs Adaptive_pro）

这个章节用于“毕设可追溯记录”：明确每一代策略到底改了什么。

### 8.1 Adaptive（第一代动态 k）相对 fixed 的改动

| 改动层面 | 具体改动 | 状态 | 证据 |
|---|---|---|---|
| 控制策略 | 引入 `_maybe_update_adaptive_k`，按 `waste/pos_accept/queue_depth` 规则动态调 k | 已确认 | 日志有 `AdaptiveK update: k x -> y` |
| 观测指标 | 在 `_run_speculative_decoding_step` 增加 `P1P2 step_accept` 和 `waste_ratio` 统计 | 已确认 | 日志有 `P1P2 k=... step_accept=... waste_ratio=...` |
| 参数化 | 用环境变量控制 `k_min/k_max/init/cooldown/thresholds` | 已确认 | 你的实验脚本已导出这些变量 |
| 工程可复现 | 固化 benchmark 脚本与 key 日志抽取 | 已确认 | `logs/server_*.log`, `logs/key_*.log` |

### 8.2 Adaptive_pro（第二代）相对 Adaptive 的改动

| 改动层面 | 具体改动 | 状态 | 证据 |
|---|---|---|---|
| 稳定性修复 | 动态 k 不再直接导致流程崩溃（此前有 `spec_token_acceptance_counts` 越界、metrics 固定 k 断言问题） | 已确认 | 当前 `adaptive_pro` 三组（2/4/8）均完整跑完 200 请求 |
| 指标兼容性 | 动态 k 下统计链路可持续输出（acc/eff/P1P2/throughput） | 已确认 | 三组 `adaptive_pro` 报告均有完整指标表 |
| 策略调优效果 | 在 `k=4` 档实现吞吐提升 + 延迟下降（相对 adaptive） | 已确认 | `+2.53%` tok/s，`TPOT -12.89%`，`ITL -12.21%` |
| 高 k 风险暴露 | `k=8` 档在 adaptive_pro 下反而显著恶化（提示高 k 门控不足） | 已确认 | `TTFT +84.62%`，`P99 ITL +145.95%`（相对 adaptive） |

### 8.3 版本结论（用于论文里的“方法演进”）

| 版本 | 一句话总结 | 当前建议 |
|---|---|---|
| fixed | 静态 k，简单但在高 k 下容易性能塌陷 | 仅作基线 |
| adaptive | 能显著改善 `k=4/8` 相对 fixed 的劣化问题 | 作为中间版本 |
| adaptive_pro | 在 `k=4` 上达到当前最佳综合点，但 `k=8` 仍需门控 | 作为当前主版本继续迭代 |

### 8.4 代码级差异对照（详细）

> 下面是“代码路径级别”的差异，按文件和函数展开，供毕设正文引用。

#### 8.4.1 `adaptive`（第一代）核心实现路径

| 文件 | 函数/位置 | 代码作用 | 关键点 |
|---|---|---|---|
| `vllm/vllm/spec_decode/spec_decode_worker.py` | `__init__`（约 357-379 行） | 初始化动态 k 参数 | 读取 `VLLM_ASCEND_ADAPTIVE_K_*`、阈值与 cooldown |
| `vllm/vllm/spec_decode/spec_decode_worker.py` | `execute_model`（约 551-573 行） | 每步设置本次 `num_lookahead_slots` | 按 `adaptive_k_current` 改写当前步 k |
| `vllm/vllm/spec_decode/spec_decode_worker.py` | `_run_speculative_decoding_step`（约 1031-1054 行） | 从 `accepted_token_ids` 计算 `step_accept` 与 `waste_ratio` | 用于驱动 k 调整与日志 |
| `vllm/vllm/spec_decode/spec_decode_worker.py` | `_maybe_update_adaptive_k`（约 726-774 行） | 阈值规则更新 k | 先降后升：高 waste/低后位接受率降档，队列深+低 waste+高 P2 升档 |

`adaptive` 规则核心（阈值版）：

```python
if k >= 4 and (waste_ratio > adaptive_waste_high or pos4 < adaptive_p4_low):
    new_k = max(k - 1, k_min)
elif k >= 3 and (waste_ratio > adaptive_waste_mid or pos3 < adaptive_p3_low):
    new_k = max(k - 1, k_min)
elif (k < k_max and queue_depth >= adaptive_queue_up
      and waste_ratio < adaptive_waste_low
      and pos2 > adaptive_p2_high):
    new_k = min(k + 1, k_max)
```

#### 8.4.2 `adaptive_pro`（第二代）新增/强化点

`adaptive_pro` 不是“另一个分支文件”，而是在 `adaptive` 基础上增加了以下代码层增强：

| 类别 | 文件 | 函数/位置 | 代码级改动 | 解决的问题 |
|---|---|---|---|---|
| 效用决策层 | `spec_decode_worker.py` | `__init__`（约 380-388 行） | 增加 utility 参数：`ADAPTIVE_ENABLE_UTILITY/EWMA_BETA/SWITCH_MARGIN/MIN_DWELL` | 阈值法抖动和局部最优 |
| 效用决策层 | `spec_decode_worker.py` | `_update_k_ewma`（约 701 行） | 维护每个 k 的 `step_accept` 与 `stage_time` 的 EWMA | 用历史统计替代单步噪声 |
| 效用决策层 | `spec_decode_worker.py` | `_estimate_expected_emitted`（约 676 行） | 估计每步期望发射 token 数 | 将 acceptance 序列映射为收益 |
| 效用决策层 | `spec_decode_worker.py` | `_estimate_utility`（约 688 行） | `utility = expected_emit / total_ms` | 显式比较“收益/耗时” |
| 效用决策层 | `spec_decode_worker.py` | `_select_k_by_utility`（约 650 行） | 用 margin + min dwell 选 k | 减少频繁抖动切换 |
| 调用链接入 | `spec_decode_worker.py` | `_run_speculative_decoding_step`（约 1066-1083 行） | 在阈值更新后再做 EWMA 更新与 utility 切换 | 从“阈值控制”升级为“阈值+效用双层控制” |
| 安全上限 | `spec_decode_worker.py` | `execute_model`（约 562-564 行） | 引入 `effective_k_max = min(adaptive_k_max, scheduled_lookahead)` | 防止 k 超出 scheduler 分配容量 |
| 输出处理稳定性 | `vllm/vllm/engine/output_processor/multi_step.py` | `_ensure_spec_acceptance_capacity`（约 82-100 行） | 动态扩容 `spec_token_acceptance_counts` | 修复 `step_index` 越界 |
| 预分配兼容 | `vllm/vllm/engine/llm_engine.py` | `_create_sequence_group_with_sampling`（约 765-773 行） | `draft_size = max(init_k, adaptive_k_max)+1` | 避免初始计数容量过小 |
| 指标兼容 | `vllm/vllm/spec_decode/metrics.py` | `get_max_num_emitted_tokens`（约 212-228 行） | fixed-k 用精确公式；dynamic-k 用保守上界（移除 assert） | 修复 `draft_tokens % k == 0` 断言崩溃 |
| 回归测试 | `vllm/tests/engine/test_multi_step_output_processor.py` | `test_spec_acceptance_counts_expand_for_dynamic_k`（约 273 行） | 新增动态 k 扩容测试 | 防止后续改动回归 |

#### 8.4.3 `adaptive` 与 `adaptive_pro` 的运行模式切换（代码级）

要做“纯阈值 adaptive（接近第一代）”：

```bash
export VLLM_ASCEND_ADAPTIVE_K_ENABLE=1
export VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY=0
```

要做“adaptive_pro（阈值 + utility）”：

```bash
export VLLM_ASCEND_ADAPTIVE_K_ENABLE=1
export VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY=1
export VLLM_ASCEND_ADAPTIVE_EWMA_BETA=0.2
export VLLM_ASCEND_ADAPTIVE_SWITCH_MARGIN=0.03
export VLLM_ASCEND_ADAPTIVE_MIN_DWELL_STEPS=12
```

#### 8.4.4 为什么把这些都归到 `adaptive_pro`

`adaptive_pro` 目标不是单点提速，而是“可跑通 + 可统计 + 可调参”的工程闭环，因此包含三层改动：

1. 控制器层：EWMA + utility 选择器。  
2. 稳定性层：scheduler cap、计数器扩容、draft_size 预分配。  
3. 指标层：dynamic-k 兼容统计，不再按 fixed-k 假设强断言。  

### 8.5 核心控制函数源码摘录与讲解（可直接引用）

> 下面代码块来自当前本地代码基线，用于论文“实现细节”章节。  
> 建议在正文中把“函数名 + 文件路径 + 作用”一起写上。

#### 8.5.1 每步 k 裁剪到 scheduler 上限（防越界）

文件：`vllm/vllm/spec_decode/spec_decode_worker.py`，`execute_model`

```python
scheduled_lookahead_slots = num_lookahead_slots

if (self.adaptive_k_enabled and not all_prompt and not disable_all_speculation
        and not all_zero_spec_tokens and num_lookahead_slots > 0):
    if self.adaptive_k_current is None:
        init_k = self.adaptive_k_init if self.adaptive_k_init > 0 else num_lookahead_slots
        self.adaptive_k_current = max(self.adaptive_k_min, min(self.adaptive_k_max, init_k))

    effective_k_max = min(self.adaptive_k_max, max(1, scheduled_lookahead_slots))
    effective_k_min = min(self.adaptive_k_min, effective_k_max)
    self._adaptive_runtime_k_cap = effective_k_max

    self.adaptive_k_current = max(
        effective_k_min,
        min(effective_k_max, self.adaptive_k_current),
    )

    num_lookahead_slots = self.adaptive_k_current
    execute_model_req.num_lookahead_slots = num_lookahead_slots
```

讲解：

- 这是动态 k 的第一道安全门，确保 `k <= scheduled_lookahead_slots`。  
- 直接避免你之前遇到的 `block_table` 越界类错误。  
- 实验日志可用 `cap=...` 验证是否生效。

#### 8.5.2 阈值控制器（adaptive 第一代核心）

文件：`vllm/vllm/spec_decode/spec_decode_worker.py`，`_maybe_update_adaptive_k`

```python
def _maybe_update_adaptive_k(self, queue_depth: int, step_accept: torch.Tensor,
                             waste_ratio: float) -> None:
    if not self.adaptive_k_enabled or self.adaptive_k_current is None:
        return

    self._adaptive_total_steps += 1
    if (self._adaptive_total_steps - self._adaptive_last_update_step
            < self.adaptive_cooldown_steps):
        return

    runtime_k_cap = self.adaptive_k_max
    if self._adaptive_runtime_k_cap is not None:
        runtime_k_cap = min(runtime_k_cap, self._adaptive_runtime_k_cap)

    k_max = max(1, runtime_k_cap)
    k_min = min(self.adaptive_k_min, k_max)
    k = max(k_min, min(k_max, self.adaptive_k_current))
    if k != self.adaptive_k_current:
        self.adaptive_k_current = k

    pos2 = self._step_accept_at(step_accept, 2, 1.0)
    pos3 = self._step_accept_at(step_accept, 3, pos2)
    pos4 = self._step_accept_at(step_accept, 4, pos3)

    new_k = k
    if k >= 4 and (waste_ratio > self.adaptive_waste_high or pos4 < self.adaptive_p4_low):
        new_k = max(k - 1, k_min)
    elif k >= 3 and (waste_ratio > self.adaptive_waste_mid or pos3 < self.adaptive_p3_low):
        new_k = max(k - 1, k_min)
    elif (k < k_max and queue_depth >= self.adaptive_queue_up
          and waste_ratio < self.adaptive_waste_low and pos2 > self.adaptive_p2_high):
        new_k = min(k + 1, k_max)

    if new_k != k:
        logger.info(
            "AdaptiveK update: k %d -> %d (waste=%.4f pos2=%.4f pos3=%.4f pos4=%.4f q=%d cap=%d)",
            k, new_k, waste_ratio, pos2, pos3, pos4, queue_depth, k_max)
        self.adaptive_k_current = new_k
        self._adaptive_last_update_step = self._adaptive_total_steps
```

讲解：

- 输入只有 3 个：`queue_depth / step_accept / waste_ratio`。  
- 策略是“先降后升”，所以在风险上更保守。  
- 你文档里的 `P1P2` 曲线与 `AdaptiveK update` 日志，正是这段代码的行为外显。

#### 8.5.3 效用控制器（adaptive_pro 增强核心）

文件：`vllm/vllm/spec_decode/spec_decode_worker.py`

```python
def _estimate_expected_emitted(self, step_accept: torch.Tensor) -> float:
    if step_accept.numel() == 0:
        return 1.0
    prod = 1.0
    expected = 1.0
    for i in range(step_accept.numel()):
        prod *= float(step_accept[i].item())
        expected += prod
    return expected

def _estimate_utility(self, k: int) -> Optional[float]:
    if k not in self._k_step_accept_ewma or k not in self._k_stage_time_ewma:
        return None
    a = self._k_step_accept_ewma[k]
    t = self._k_stage_time_ewma[k]
    expected_emit = self._estimate_expected_emitted(a)
    total_ms = float(t[0].item()) * k + float(t[1].item()) + float(t[2].item())
    if total_ms <= 1e-6:
        return None
    return expected_emit / total_ms

def _select_k_by_utility(self, current_k: int, k_min: int, k_max: int) -> int:
    if not self.adaptive_enable_utility:
        return current_k
    if self._adaptive_total_steps - self._adaptive_last_switch_step < self.adaptive_min_dwell_steps:
        return current_k

    best_k = current_k
    best_u = self._estimate_utility(current_k)
    if best_u is None:
        return current_k

    for cand_k in range(k_min, k_max + 1):
        if cand_k == current_k:
            continue
        u = self._estimate_utility(cand_k)
        if u is None:
            continue
        if u > best_u * (1.0 + self.adaptive_switch_margin):
            best_k, best_u = cand_k, u

    if best_k != current_k:
        self._adaptive_last_switch_step = self._adaptive_total_steps
    return best_k
```

讲解：

- 与纯阈值不同，`adaptive_pro` 多了一层“收益/耗时”比较。  
- `adaptive_switch_margin` + `adaptive_min_dwell_steps` 是抗抖动关键。  
- 你看到 `k=8` 退化，说明目前效用函数对尾时延惩罚还不够强（后续优化点）。

#### 8.5.4 P1P2 指标与控制器触发点

文件：`vllm/vllm/spec_decode/spec_decode_worker.py`，`_run_speculative_decoding_step`

```python
spec_indices = (proposals.proposal_lens > 0).nonzero(as_tuple=False).squeeze(-1)
if spec_indices.numel() > 0:
    spec_indices = spec_indices.tolist()
    accepted_spec = (accepted_token_ids[spec_indices, :-1] != -1)  # [n_spec, k]
    step_accept = accepted_spec.float().mean(dim=0)                 # [k]

    accepted_cnt = accepted_spec.sum().item()
    proposed_cnt = accepted_spec.numel()
    waste_ratio = 1.0 - accepted_cnt / max(proposed_cnt, 1)

    self._maybe_update_adaptive_k(
        queue_depth=execute_model_req.running_queue_size,
        step_accept=step_accept,
        waste_ratio=waste_ratio,
    )

    self._update_k_ewma(
        k=num_lookahead_slots,
        step_accept=step_accept,
        stage_times=(proposal_timer.elapsed_time_ms / max(num_lookahead_slots, 1),
                     scoring_timer.elapsed_time_ms,
                     verification_timer.elapsed_time_ms),
    )
```

讲解：

- `step_accept` 是位置接受率（P1/P2/P3...），`waste_ratio` 是草稿浪费率。  
- 阈值控制器先执行，再由 EWMA/utility 做二次决策。  
- 因此日志里通常同时出现：`P1P2 ...`、`AdaptiveK update ...`、`AdaptiveK utility switch ...`。

#### 8.5.5 dynamic-k 指标兼容修复（metrics）

文件：`vllm/vllm/spec_decode/metrics.py`

```python
@staticmethod
def get_max_num_emitted_tokens(draft_tokens: int,
                               k: int,
                               min_k_observed: Optional[int] = None,
                               dynamic_k_seen: bool = False) -> int:
    if draft_tokens <= 0:
        return 0

    # Fixed-k exact path
    if (not dynamic_k_seen) and k > 0 and draft_tokens % k == 0:
        total_num_spec_seqs = draft_tokens // k
        return total_num_spec_seqs * (k + 1)

    # Dynamic-k fallback (no assert)
    k_floor = min_k_observed if (min_k_observed is not None and min_k_observed > 0) else max(k, 1)
    total_num_spec_seqs_upper = (draft_tokens + k_floor - 1) // k_floor
    return draft_tokens + total_num_spec_seqs_upper
```

讲解：

- 这是修复 `assert draft_tokens % k == 0` 崩溃的关键。  
- fixed-k 仍走精确路径，dynamic-k 走保守上界路径。  
- 这保证了 `system_efficiency` 在动态 k 下可持续统计。

#### 8.5.6 输出处理扩容修复（multi_step）

文件：`vllm/vllm/engine/output_processor/multi_step.py`

```python
@staticmethod
def _ensure_spec_acceptance_capacity(sequence_group: SequenceGroup,
                                     step_index: int) -> int:
    if step_index is None:
        step_index = 0
    if step_index < 0:
        step_index = 0

    counts = sequence_group.metrics.spec_token_acceptance_counts
    if counts is None:
        sequence_group.metrics.spec_token_acceptance_counts = [0] * (step_index + 1)
        return step_index

    if step_index >= len(counts):
        counts.extend([0] * (step_index + 1 - len(counts)))

    return step_index
```

讲解：

- 解决 dynamic-k 提高后 `step_index` 超过原长度导致的 `IndexError`。  
- 对应回归测试：`test_spec_acceptance_counts_expand_for_dynamic_k`。

---

## 9. 本轮已验证的常见误区

1. 把 `num_speculative_tokens` 当“初始值”而非“容量上限”。  
2. 动态 k 超过 scheduler 当步 lookahead，导致 NPU attention 越界。  
3. 动态 k 还沿用固定 k 指标假设，触发 metrics 断言。  
4. 只看单次结果，不看重复实验和稳态窗口。  



