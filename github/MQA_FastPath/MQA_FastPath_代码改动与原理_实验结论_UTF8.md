# MQA 与 FastPath（GPU multi-step）代码改动与原理说明（2026-03-11 提取表覆盖版）

## 1. 文档范围说明

这版文档做了两件事：
- 保留并细化“代码改动 + 原理”讲解（不删代码原理部分）。
- 用 `github/MQA_FastPath/extracted_tables_20260311_234131` 的提取表覆盖实验数据与结论。

---

## 2. 数据来源（本次覆盖使用）

提取目录：`github/MQA_FastPath/extracted_tables_20260311_234131`

核心文件：
- `TABLE_SUMMARY.md`
- `summary_combined.csv`
- `bench_metrics.csv`
- `master_run_table.csv`
- `log_key_values.csv`
- `log_last_values.csv`
- `all_log_lines.csv`

`TABLE_SUMMARY.md` 记录的生成时间：`2026-03-11 23:42:08`。

### 2.1 文件规模概览

| data_group | file_count | log_count | csv_count | total_size_bytes |
|---|---:|---:|---:|---:|
| A&B | 27 | 26 | 1 | 4,233,477 |
| C&D | 27 | 26 | 1 | 4,157,914 |
| no_spec | 6 | 5 | 1 | 643,248 |

### 2.2 统一实验配置（来自 meta/probe）

- Target: `Qwen3-8B`
- Draft: `Qwen3-0.6B`
- Dataset: `ShareGPT_V3_unfiltered_cleaned_split.json`
- `num_prompts=200`
- `request_rate=16`
- `max_model_len=8192`
- `enforce_eager=True`
- `VLLM_USE_V1=0`

注：`summary_combined.csv` 含 1 行历史失败记录（`A&B / C_FP_OFF_MQA_ON / k=2 / SERVER_FAIL`），统计对比时已按 `status=OK` 过滤。

---

## 3. 代码与原理（详细）

### 3.1 vLLM 原始控制流（为什么会回退）

#### 3.1.1 FastPath 入口

文件：`vllm/vllm/spec_decode/multi_step_worker.py`

核心判定（语义）：
- 必须是 `TP1DraftModelRunner`。
- 必须 `supports_gpu_multi_step(expanded_request) == True`。
- 否则进入 CPU prepare/fallback 路径。

原理：
- FastPath 的本质是把 draft proposal 的“多步准备+执行”尽量放在设备侧连续推进。
- 任一前置条件不满足，就会回到逐步 prepare（CPU 参与更多）的通路。

#### 3.1.2 Scorer 选择

文件：`vllm/vllm/spec_decode/spec_decode_worker.py`

核心分支（语义）：
- `disable_mqa_scorer=True` -> `BatchExpansionTop1Scorer`
- 否则 -> `MQAScorer`

原理：
- MQA scorer 目标是降低 scoring 阶段的重复计算/扩批成本。
- 如果能力判定过严或误判，就会被迫走 batch expansion。

---

### 3.2 Ascend 侧改动：MQA 能力判定

文件：`vllm-ascend/vllm_ascend/patch/worker/patch_common/patch_spec_decode_worker.py`

改动点：
1. 从“后端名字硬编码”升级为“能力接口优先 + allowlist 兜底”。
2. 保留安全约束（例如长度约束、eager 约束），避免误开。
3. 增强日志：
   - `MQA scorer backend capability detected ...`
   - `Use MQA scorer for scoring proposals.`
   - `Use batch expansion for scoring proposals.`

原理：
- 先问“能不能”（capability）而不是“你叫什么名字”（backend 名）。
- 这样 Ascend 即使不是传统 FLASH_ATTN 命名，也能按能力启用 MQA。

---

### 3.3 Ascend 侧改动：GPU multi-step FastPath 判定与探针

文件：
- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`
- `vllm-ascend/vllm_ascend/patch/worker/patch_common/patch_multi_step_worker.py`

改动点：
1. `supports_gpu_multi_step` 引入可解释判定（含后端能力、metadata 能力、LoRA/adapter 条件）。
2. 增加失败原因计数（`fail_backend/fail_prompt/...`）。
3. 在 multi-step 决策点打印探针：
   - `supports_gpu_multi_step=True/False ...`
   - `fallback CPU prepare because supports_gpu_multi_step=False.`

原理：
- 不是只要“能跑”就够，必须能解释“为什么没跑”。
- `supports_gpu_multi_step=False + fail_backend=1` 可以直接定位是后端能力侧回退。

---

### 3.4 你点名的阶段耗时采集（关键）

你关心的是两层：

1. Worker 调度/准备阶段（`worker_base.py:464`）
   - `prepare_worker_input_ms`
   - `prepare_model_input_ms`
   - `execute_worker_ms`
   - `model_execute_ms`

2. 模型执行内部阶段（`model_runner.py:1433` 附近）
   - `forward_ms`
   - `logits_ms`
   - `sampler_ms`

当前代码状态：
- `vllm-ascend/vllm_ascend/worker/model_runner.py` 已有聚合日志：
  - `BaseRunner stage times: avg_forward_ms=... avg_logits_ms=... avg_sampler_ms=... avg_execute_total_ms=...`
- worker 侧你要的四段字段目前在本轮提取表中仍未出现（第 7 节给证据）。

---

## 4. 命中证据矩阵（按提取日志判定）

| 组别 | scorer 路径 | multi-step 判定 | fallback |
|---|---|---|---|
| A (`FP_ON+MQA_ON`) | `Use MQA scorer` | `supports_gpu_multi_step=True` | 无 |
| B (`FP_ON+MQA_OFF`) | `Use batch expansion` | `supports_gpu_multi_step=True` | 无 |
| C (`FP_OFF+MQA_ON`) | `Use MQA scorer` | `supports_gpu_multi_step=False` + `fail_backend=1` | 有 |
| D (`FP_OFF+MQA_OFF`) | `Use batch expansion` | `supports_gpu_multi_step=False` + `fail_backend=1` | 有 |
| NO_SPEC | 不适用 | 不适用 | 不适用 |

说明：`all_log_lines.csv` 中未检索到 `DraftGPUFastPath` 字段；本轮以 `MultiStep probe` 作为 FastPath 命中/回退证据主字段。

---

## 5. 实验结果（提取表覆盖）

### 5.1 吞吐与时延（summary_combined.csv, status=OK）

| 组别 | k | Throughput (tok/s) | Mean TPOT (ms) | Mean ITL (ms) | mqa_mode | supports_gpu_multi_step |
|---|---:|---:|---:|---:|---|---|
| A | 2 | 513.78 | 307.19 | 311.45 | MQA | True |
| A | 4 | 519.02 | 326.97 | 340.11 | MQA | True |
| A | 8 | 542.98 | 302.78 | 343.23 | MQA | True |
| B | 2 | 327.65 | 303.48 | 314.90 | BATCH_EXPANSION | True |
| B | 4 | 314.75 | 352.96 | 332.24 | BATCH_EXPANSION | True |
| B | 8 | 328.03 | 316.49 | 319.65 | BATCH_EXPANSION | True |
| C | 2 | 473.06 | 316.05 | 334.03 | MQA | False |
| C | 4 | 493.59 | 312.44 | 358.60 | MQA | False |
| C | 8 | 524.13 | 330.46 | 381.50 | MQA | False |
| D | 2 | 400.65 | 263.80 | 375.08 | BATCH_EXPANSION | False |
| D | 4 | 403.54 | 252.10 | 396.68 | BATCH_EXPANSION | False |
| D | 8 | 374.72 | 259.96 | 452.66 | BATCH_EXPANSION | False |
| NO_SPEC | 0 | 706.73 | 139.53 | 92.72 | UNKNOWN | - |

注：提取表中 C/D 的 `supports_gpu_multi_step` 原值为 `False.`（带句点），文档中按布尔语义归一化写为 `False`。

### 5.2 消融增益（以 throughput_tps 计算）

| k | A vs B（同 FastPath 比 MQA） | C vs D（同回退比 MQA） | A vs C（同 MQA 比 FastPath） | B vs D（同 BatchExp 比 FastPath） |
|---:|---:|---:|---:|---:|
| 2 | +56.81% | +18.07% | +8.61% | -18.22% |
| 4 | +64.90% | +22.32% | +5.15% | -22.00% |
| 8 | +65.53% | +39.87% | +3.60% | -12.46% |

三点结论：
- MQA 在 ON/OFF 两侧都明显提高吞吐（A>B，C>D）。
- FastPath 在 MQA ON 条件下有稳定正收益（A>C）。
- 在 BatchExpansion 路径下 FastPath 不一定带来端到端收益（B<D）。

### 5.3 与 no-spec 的差距（bench_metrics.csv）

- no-spec：`706.73 tok/s`
- 最优 spec：`A_k8 = 542.98 tok/s`，较 no-spec 低 `23.17%`
- A 组均值：`525.26 tok/s`，较 no-spec 低 `25.68%`
- 全部 spec 均值：`434.66 tok/s`，较 no-spec 低 `38.50%`

补充（A_k8 vs no-spec）：

| 指标 | A_k8 | no-spec |
|---|---:|---:|
| mean_ttft_ms | 260.16 | 231.40 |
| p99_ttft_ms | 664.14 | 693.38 |
| mean_tpot_ms | 302.78 | 139.53 |
| mean_itl_ms | 343.23 | 92.72 |
| p99_itl_ms | 2224.19 | 402.16 |

解读：
- spec 的主要差距仍在 token 级时延（TPOT/ITL），尤其尾部 ITL。
- 这说明除了 scorer/FastPath 命中，调度与额外链路抖动仍是核心差距来源。

### 5.4 模型执行内部阶段（log_last_values 聚合）

| 组别 | avg_execute_total_ms | avg_forward_ms | avg_logits_ms | avg_sampler_ms |
|---|---:|---:|---:|---:|
| A（k=2/4/8均值） | 67.70 | 61.99 | 1.40 | 4.26 |
| B（k=2/4/8均值） | 67.81 | 63.36 | 1.09 | 3.30 |
| C（k=2/4/8均值） | 69.57 | 63.63 | 1.49 | 4.41 |
| D（k=2/4/8均值） | 70.09 | 63.94 | 1.35 | 4.75 |
| NO_SPEC | 60.40 | 49.99 | 0.63 | 3.24 |

---

## 6. 你关心的字段：当前是否“在提取数据里”

| 字段 | 你给的位点 | 当前提取结果 | 证据 |
|---|---|---|---|
| `prepare_worker_input_ms` | `worker_base.py:464` | 未看到 | `all_log_lines.csv` 无匹配 |
| `prepare_model_input_ms` | `worker_base.py:464` | 未看到 | 同上 |
| `execute_worker_ms` | `worker_base.py:464` | 未看到 | 同上 |
| `model_execute_ms` | `worker_base.py:464` | 未看到 | 同上 |
| `forward_ms` | `model_runner.py:1433` 附近 | 有（聚合） | `avg_forward_ms` |
| `logits_ms` | `model_runner.py:1476` 附近 | 有（聚合） | `avg_logits_ms` |
| `sampler_ms` | `model_runner.py:1501` 附近 | 有（聚合） | `avg_sampler_ms` |

一句话：
- 模型内部三段本轮有数据。
- Worker 调度/准备四段在这批提取表里仍缺失，需要补日志点并重跑。

---

## 7. 当前可确认结论（基于 extracted_tables）

### 7.1 贡献度拆解：谁是主因、谁是次因

- MQA 是主收益来源。
  - 同 FastPath 条件下（A vs B），k=2/4/8 分别 +56.81% / +64.90% / +65.53%，均值 +62.41%。
  - 同回退条件下（C vs D），k=2/4/8 分别 +18.07% / +22.32% / +39.87%，均值 +26.75%。
- FastPath 是次收益来源，而且依赖 scorer 路径。
  - MQA ON 下（A vs C），k=2/4/8 为 +8.61% / +5.15% / +3.60%，均值 +5.79%。
  - MQA OFF 下（B vs D）反向为 -18.22% / -22.00% / -12.46%，均值 -17.56%。
- 结论：在你这套 Qwen3-8B/0.6B + 当前负载下，先保证 MQA 命中，再谈 FastPath 才有稳定收益。

### 7.2 为什么“优化后 spec”仍明显落后 no-spec

- 吞吐差距：最优 spec（A_k8）542.98 tok/s，仍比 no-spec 706.73 tok/s 低 23.17%。
- token 时延差距更大：
  - mean TPOT: 302.78 vs 139.53（2.17x）
  - mean ITL: 343.23 vs 92.72（3.70x）
  - p99 ITL: 2224.19 vs 402.16（5.53x）
- 模型执行内部也更重（A 组均值 vs no-spec）：
  - avg_execute_total_ms: 67.70 vs 60.40（+12.09%）
  - avg_forward_ms: 61.99 vs 49.99（+24.00%）
- 说明：瓶颈不只是“有没有命中 MQA/FastPath”，而是 speculative 的额外链路（proposal/scoring/verification/调度）叠加后，把 token 级时延尤其尾时延拉高了。

### 7.3 你这批数据支持的机理解释（可写入论文）

1. MQA 主要降低 scoring 开销，因此直接拉升 throughput。
2. FastPath 主要降低 draft proposal 的 prepare/推进成本，但其收益幅度小于 scorer 路径选择。
3. 当 scorer 走 batch expansion 时，scoring 与调度开销会吞掉 FastPath 的收益（因此 B<D）。
4. 当前 workload（ShareGPT 混合长度 + request_rate=16）下，spec 尚未达到 break-even。
   - 可用 break-even 条件表述：`Spec收益 > proposal + scoring + verification + prepare额外开销`。

### 7.4 证据链完整性与当前缺口

- 已有硬证据：
  - MQA 命中日志：`Use MQA scorer ...`
  - FastPath 命中/回退日志：`supports_gpu_multi_step=True/False` + `fail_backend` + `fallback CPU prepare`
- 当前缺口：
  - `prepare_worker_input_ms / prepare_model_input_ms / execute_worker_ms / model_execute_ms` 仍缺失，导致无法把“CPU prepare vs 模型执行”做严格定量拆分。
- 这也是目前“结论能解释趋势，但还不能精确归因到每一步毫秒数”的根本原因。

---

## 8. 下一步建议（针对你毕业设计可直接落地）

1. 先补齐 `worker_base.py:464` 的四段日志（字段名保持你现在的命名），再重跑 A/B/C/D + no-spec，形成完整耗时分解表。
2. 把实验按输入长度分桶（短/中/长）并固定并发，分别统计 P50/P95/P99 ITL，验证尾时延是否主要来自长上下文与调度抖动。
3. 额外记录 acceptance/waste（如 step_accept、waste_ratio），建立“接受率 -> 吞吐收益”的拟合曲线，确定 spec 的 break-even 区间。
4. 论文中建议采用“主效应（MQA）+ 次效应（FastPath）+ 交互项（scorer x fastpath）”的消融叙述，避免只给单一平均值结论。