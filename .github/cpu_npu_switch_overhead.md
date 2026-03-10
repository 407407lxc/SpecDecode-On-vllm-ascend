# NPU↔CPU 切换损耗优化（指标明确版）

> 代码对齐日期：2026-03-07

## 1. 方向目标

减少 spec-decode 多步过程中的 Host/Device 往返与 Python 化开销，提升吞吐并降低尾延迟。

---

## 2. 统一指标与 Baseline（throughput / latency / acceptance）

## 2.1 指标口径

- `throughput_tps = accepted_tokens_total / wall_time_seconds`
- `iter_latency_ms_p50/p95 = spec step wall time 分位数`
- `accept_rate = accepted_tokens_total / proposed_tokens_total`

CPU/NPU 切换专属补充指标：

- `gpu_multi_step_hit_rate = ms_hit / ms_total`
- `d2h_bytes_per_step`
- `pythonize_ms_per_step`

## 2.2 Baseline 设计（必须先跑）

1. `C0-K1`：`k=1`（近似回退路径），logprobs 关闭。  
2. `C1-K8-LogprobsOn`：`k=8`，logprobs 开启。  
3. `C2-K8-LogprobsOff`：`k=8`，logprobs 关闭。

```bash
# C0/C2
export VLLM_LOGPROBS_MODE=off
# C1
# 通过请求 sampling params 打开 logprobs
```

每个 baseline 固定填表：

| Baseline | throughput_tps | iter_latency_ms_p95 | accept_rate | gpu_multi_step_hit_rate | pythonize_ms_per_step | d2h_bytes_per_step |
|---|---:|---:|---:|---:|---:|---:|
| C0-K1 |  |  |  |  |  |  |
| C1-K8-LogprobsOn |  |  |  |  |  |  |
| C2-K8-LogprobsOff |  |  |  |  |  |  |

---

## 3. 检验问题实验（Problem Revelation）

## 3.1 实验 P1：快路命中率与失败原因分布

目标：证明大量请求未命中 `supports_gpu_multi_step`，并量化对吞吐/延迟/接受率的影响。

```python
# __init__
self.ms_total = 0
self.ms_hit = 0
self.ms_fail_prompt_mode = 0
self.ms_fail_attn_backend = 0
self.ms_fail_lora = 0
self.ms_fail_prompt_adapter = 0

self.metric_start_ts = time.perf_counter()
self.metric_proposed_tokens = 0
self.metric_accepted_tokens = 0
self.metric_iter_latency_ms = []


def _pctl(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, int((pct / 100.0) * (len(arr) - 1)))
    return float(arr[idx])


# in TP1DraftModelRunner.supports_gpu_multi_step
self.ms_total += 1
if seq_group.is_prompt:
    self.ms_fail_prompt_mode += 1
    return False
...
self.ms_hit += 1
return True

# in decode loop, after _verify_tokens
proposed = accepted_token_ids.shape[0] * num_lookahead_slots
accepted = (accepted_token_ids[:, :-1] != -1).sum().item()
self.metric_proposed_tokens += proposed
self.metric_accepted_tokens += accepted
self.metric_iter_latency_ms.append(iter_ms)

if self.ms_total % 200 == 0 and self.ms_total > 0:
    hit_rate = self.ms_hit / self.ms_total
    accept_rate = self.metric_accepted_tokens / max(self.metric_proposed_tokens, 1)
    elapsed_s = max(time.perf_counter() - self.metric_start_ts, 1e-6)
    throughput_tps = self.metric_accepted_tokens / elapsed_s
    p95_ms = _pctl(self.metric_iter_latency_ms, 95)
    logger.info(
        "P1 hit_rate=%.4f throughput_tps=%.3f iter_p95=%.3f accept=%.4f fail_prompt=%d fail_backend=%d fail_lora=%d fail_adapter=%d",
        hit_rate,
        throughput_tps,
        p95_ms,
        accept_rate,
        self.ms_fail_prompt_mode,
        self.ms_fail_attn_backend,
        self.ms_fail_lora,
        self.ms_fail_prompt_adapter,
    )
```

预期：至少一个 baseline 中 `hit_rate` 偏低，且存在主导失败原因。

## 3.2 实验 P2：Host 传输与 pythonization 成本

目标：验证 `logprobs` 与 CPU 传输造成的延迟占比，并观察对三主指标的拖累。

```python
# __init__
self.total_d2h_bytes = 0
self.total_pythonize_ms = 0.0
self.total_steps = 0

# pseudo profiler counters in multi-step loop
iter_t0 = time.perf_counter()
outputs = worker.execute_model(...)
iter_ms = (time.perf_counter() - iter_t0) * 1000.0

# 这些计数器来自 multi_step_runner/deferred pythonization 路径
self.total_d2h_bytes += d2h_bytes_counter
self.total_pythonize_ms += pythonize_time_ms
self.total_steps += 1
self.metric_iter_latency_ms.append(iter_ms)

if self.total_steps % 200 == 0:
    accept_rate = self.metric_accepted_tokens / max(self.metric_proposed_tokens, 1)
    elapsed_s = max(time.perf_counter() - self.metric_start_ts, 1e-6)
    throughput_tps = self.metric_accepted_tokens / elapsed_s
    p95_ms = _pctl(self.metric_iter_latency_ms, 95)
    logger.info(
        "P2 throughput_tps=%.3f iter_p95=%.3f accept=%.4f d2h_bytes_per_step=%.1f pythonize_ms_per_step=%.3f",
        throughput_tps,
        p95_ms,
        accept_rate,
        self.total_d2h_bytes / self.total_steps,
        self.total_pythonize_ms / self.total_steps,
    )
```

预期：`C1-K8-LogprobsOn` 的 `pythonize_ms_per_step` 与 `d2h_bytes_per_step` 显著高于 `C2-K8-LogprobsOff`。

## 3.3 基线测试命令

```powershell
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_multi_step_worker.py -q
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_spec_decode_worker.py -q
```

---

## 4. 现有代码里已可利用的机制

## 4.1 TP1 draft 快路（V0）

文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

- decode 且 `num_steps>1` 时设置 `skip_sampler_cpu_output=True`。  
- `_gpu_advance_step` 在设备侧更新下一步输入。  
- `reuse_sampling_tensors=True` 减少重复构建开销。

## 4.2 Multi-step 延迟 pythonize

文件：`vllm-ascend/vllm_ascend/worker/multi_step_runner.py`

- 多步执行时强制 `skip_sampler_cpu_output=True`。  
- 中间步清空 `sampled_token_ids/probs/logprobs`，避免中途 D2H。  
- 仅在必要时做 deferred pythonization。

---

## 5. 最小可落地改动清单

## 5.1 增加“快路未命中原因”统计（必做）

实现文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

在 `supports_gpu_multi_step` 增加：

- `reason_prompt_mode`
- `reason_attn_backend`
- `reason_lora`
- `reason_prompt_adapter`

并周期打印：

- `gpu_multi_step_hit_rate`
- `throughput_tps`
- `iter_latency_ms_p95`
- `accept_rate`

## 5.2 减少非必要 logprobs 返回（必做）

执行策略：

- 吞吐基准默认关闭 logprobs。  
- 仅质量实验开启 logprobs。  
- 对比 C1/C2 时必须同时报告三主指标。

## 5.3 bonus token 路径张量化（可选）

实现文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

把逐序列写回改为批量 index 操作，减少 Python 循环和潜在同步。

## 5.4 最小补丁示例（代码块）

```diff
*** a/vllm-ascend/vllm_ascend/worker/draft_model_runner.py
--- b/vllm-ascend/vllm_ascend/worker/draft_model_runner.py
@@ class TP1DraftModelRunner:
      def __init__(self, model_runner):
          super().__init__(model_runner)
+         self.ms_total = 0
+         self.ms_hit = 0
+         self.ms_fail_prompt_mode = 0
+         self.ms_fail_attn_backend = 0
+         self.ms_fail_lora = 0
+         self.ms_fail_prompt_adapter = 0
@@
      def supports_gpu_multi_step(self, execute_model_req):
+         self.ms_total += 1
          if not allow_gpu_advance_step:
              return False
          ...
+         self.ms_hit += 1
+         return True
```

```python
# periodic logging
if self.ms_total % 200 == 0 and self.ms_total > 0:
    logger.info(
        "gpu_multi_step_hit_rate=%.3f throughput_tps=%.3f iter_p95=%.3f accept=%.4f total=%d hit=%d fail_prompt=%d fail_backend=%d fail_lora=%d fail_adapter=%d",
        self.ms_hit / self.ms_total,
        throughput_tps,
        p95_ms,
        accept_rate,
        self.ms_total,
        self.ms_hit,
        self.ms_fail_prompt_mode,
        self.ms_fail_attn_backend,
        self.ms_fail_lora,
        self.ms_fail_prompt_adapter,
    )
```

---

## 6. 验收标准

1. `C2-K8-LogprobsOff` 相比 `C1-K8-LogprobsOn`：`pythonize_ms_per_step` 与 `d2h_bytes_per_step` 显著下降。  
2. 快路命中率相比基线有明确改善（建议 `+15%` 以上）。  
3. `throughput_tps` 提升 `>= 5%`，且 `iter_latency_ms_p95` 不恶化超过 `3%`。  
4. `accept_rate` 不下降，或下降不超过 `1` 个百分点且吞吐收益显著。

---

## 7. 与动态步长联动

该方向与 `dynamic_step_adjustment.md` 强耦合：

- `k` 高：快路收益更大，但接受率下滑风险增加。  
- `k` 低：切换开销占比升高。

建议最终采用“动态 `k` + 快路命中率治理”联合策略，并统一汇报三主指标。

---

## 8. 自动化执行脚本（新增）

仓库新增：`scripts/cpu_overhead_bench.sh`

用途：直接运行 C0/C1/C2 三组对照，并自动抽取以下指标到 `summary.csv`：

- `output_tok_s` / `total_tok_s`
- `mean_tpot_ms` / `mean_itl_ms` / `p99_tpot_ms` / `p99_itl_ms`
- `draft_acceptance_rate` / `system_efficiency`
- `DraftGPUFastPath` 命中率与失败原因计数（`fail_prompt/fail_backend/fail_lora/fail_adapter`）

示例：

```bash
bash scripts/cpu_overhead_bench.sh \
  RUN_C0=1 RUN_C1=1 RUN_C2=1 \
  C0_K=1 C1_K=8 C2_K=8 \
  REQUEST_RATE=16 NUM_PROMPTS=200
```
