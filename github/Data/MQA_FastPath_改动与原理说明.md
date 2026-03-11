# MQA 与 GPU Multi-Step FastPath 改动与原理说明

## 1. 目标

本文档说明两件事：

1. MQA scorer 为什么会开启/关闭。
2. draft 的 GPU multi-step fastpath 为什么会开启/关闭。

并明确区分：

- `vllm`：基线逻辑（关键路径说明）。
- `vllm-ascend`：本次实际改动（能力判定 + 日志探针，不改算法）。

---

## 2. vLLM 基线关键路径（未直接改源码）

### 2.1 MultiStepWorker 的 fastpath 分支条件

文件：`vllm/vllm/spec_decode/multi_step_worker.py`

关键逻辑（行 88-90）：

```python
if current_platform.is_cuda_alike() and isinstance(
        self.model_runner, TP1DraftModelRunner
) and self.model_runner.supports_gpu_multi_step(expanded_request):
```

含义：

- 平台必须是 `cuda_alike`。
- draft runner 必须是 `TP1DraftModelRunner`。
- `supports_gpu_multi_step(...)` 返回 True。

否则走 CPU prepare 的逐步循环（行 98 之后）。

### 2.2 scorer 选择（MQA vs batch expansion）

文件：`vllm/vllm/spec_decode/spec_decode_worker.py`

关键逻辑（行 472-479）：

```python
if self.disable_mqa_scorer:
    scorer_cls = BatchExpansionTop1Scorer
    logger.info("[Speculative Decoding] Use batch expansion for scoring proposals.")
else:
    scorer_cls = MQAScorer
    logger.info("[Speculative Decoding] Use MQA scorer for scoring proposals.")
```

含义：

- `disable_mqa_scorer=False` 才会用 MQA。
- 否则回退为 batch expansion。

---

## 3. vllm-ascend 改动点（本次实际代码改动）

### 3.1 `patch_spec_decode_worker.py`：MQA 后端能力判定改造

文件：`vllm-ascend/vllm_ascend/patch/worker/patch_common/patch_spec_decode_worker.py`

#### 改动 A：从“硬编码 FLASH_ATTN”改为“能力函数 + allowlist”

关键逻辑（行 59-70）：

```python
supports_fn = getattr(attn_backend, "supports_spec_mqa_scorer", None)
...
allowlist = _read_backend_allowlist_env(
    "VLLM_ASCEND_SPEC_MQA_BACKENDS", "FLASH_ATTN,ASCEND")
supports = backend_name in allowlist
```

说明：

- 优先调用后端能力函数 `supports_spec_mqa_scorer()`。
- 若无函数，回退到环境变量 allowlist：
  - `VLLM_ASCEND_SPEC_MQA_BACKENDS`
  - 默认 `FLASH_ATTN,ASCEND`

#### 改动 B：明确日志

关键逻辑（行 179-193）：

```python
if not supports_mqa:
    logger.info("Disabling MQA scorer as the backend capability is unavailable ...")
else:
    logger.info("MQA scorer backend capability detected ...")
```

说明：

- 现在可以直接从日志看出“后端能力是否识别成功”。

#### 改动 C：补充 TP1DraftModelRunner 注入日志

关键逻辑（行 124-140）：

```python
draft_worker_kwargs["model_runner_cls"] = TP1DraftModelRunner
logger.info("Draft runner_cls=%s ...")
```

说明：

- 明确记录 draft runner 是否被设置为 `TP1DraftModelRunner`。

---

### 3.2 `draft_model_runner.py`：fastpath 能力判定与原因计数

文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

#### 改动 A：后端能力判定支持能力函数 + allowlist

关键逻辑（行 130-148）：

```python
supports_fn = getattr(backend, "supports_spec_gpu_multi_step", None)
...
return self._get_attn_backend_name() in self._gpu_multistep_backend_allowlist
```

并增加 metadata 能力校验：

```python
return hasattr(metadata_cls, "advance_step")
```

说明：

- 不再死绑某个元数据类型；只要 metadata 提供 `advance_step` 即可。

#### 改动 B：fastpath 统计计数 + 日志

关键逻辑（行 71-87、151-176）：

- 计数器：`ms_total/ms_hit/fail_disabled/fail_prompt/fail_backend/fail_lora/fail_adapter`
- 日志：
  - `DraftGPUFastPath allow_backends=... backend=...`
  - `DraftGPUFastPath hit_rate=... fail_backend=...`

#### 改动 C：`supports_gpu_multi_step` 细粒度失败原因

关键逻辑（行 431-466）：

- 判定顺序：
  - 全局开关
  - decode-only（不能含 prompt）
  - 后端能力 + metadata.advance_step
  - lora
  - prompt_adapter

说明：

- 可以明确看到失败落在 `prompt/backend/lora/adapter` 哪一类。

---

### 3.3 `patch_multi_step_worker.py`：调用链探针（证明 fastpath 真的被调用）

文件：`vllm-ascend/vllm_ascend/patch/worker/patch_common/patch_multi_step_worker.py`

关键逻辑（行 53-107）：

```python
logger.info("MultiStep probe: model_runner=%s is_tp1_runner=%s sample_len=%d", ...)
...
supports_gpu_multi_step = self.model_runner.supports_gpu_multi_step(expanded_request)
logger.info("MultiStep probe: supports_gpu_multi_step=%s ms_total=%s ms_hit=%s ...", ...)
...
if is_tp1_runner and supports_gpu_multi_step:
    # GPU multi-step path
else:
    # CPU prepare fallback
```

说明：

- 即使 `draft_model_runner` 的日志因为 logger 路由不可见，也能从该探针证明分支命中。

---

## 4. 运行时判定顺序（简化）

1. `patch_spec_decode_worker.create_worker` 决定 draft runner 是否为 `TP1DraftModelRunner`。
2. `patch_spec_decode_worker.create_worker` 决定 MQA 能否启用。
3. `patch_multi_step_worker.sampler_output` 调用 `supports_gpu_multi_step`。
4. 若 True，进入 GPU multi-step；若 False，回退 CPU prepare。

---

## 5. 如何从日志证明“不是只有 MQA 生效”

你已观测到：

- `Use MQA scorer for scoring proposals.`
- `MultiStep probe: ... is_tp1_runner=True ...`
- `MultiStep probe: supports_gpu_multi_step=True ms_hit=1 fail_*=0`

这三条组合已经能证明：

1. MQA scorer 生效。
2. fastpath 条件判定生效且返回 True。
3. MultiStepWorker 的执行分支满足 GPU path 条件（不是仅 MQA）。

结论：当前优化结果是“MQA + fastpath”共同作用，而不是单独 MQA。

---

## 6. 关键环境变量

- `VLLM_ASCEND_SPEC_MQA_BACKENDS`：MQA 后端 allowlist。
- `VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS`：fastpath 后端 allowlist。
- `VLLM_ASCEND_MS_STATS_INTERVAL`：fastpath 统计日志间隔。
- `VLLM_ASCEND_ADAPTIVE_K_ENABLE`：建议消融时设为 `0`，避免自适应 K 干扰。

---

## 7. 消融建议（用于论文结论）

固定同一负载，做 2x2：

- A: FP_ON + MQA_ON
- B: FP_ON + MQA_OFF
- C: FP_OFF + MQA_ON
- D: FP_OFF + MQA_OFF

主要比较：

- `Output token throughput`
- `Mean TPOT`
- `SpecDecodeWorker stage times` 中 `scoring_time_ms`

解释方式：

- `A-B` 近似 MQA 贡献（FP 固定 ON）。
- `A-C` 近似 fastpath 贡献（MQA 固定 ON）。
