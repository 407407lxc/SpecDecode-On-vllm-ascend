# NPU 端 Spec-Decode 代码总览与落地入口（V0/V1）

> 代码对齐日期：2026-03-07

本文件是 `.github` 文档的总入口，合并了原 `spec_decode_v0.md` 与 `doc_review_notes.md` 的核心内容。

## 1. 先给结论（避免走错路径）

1. 通用 `draft_model` 在 NPU 上主要走 **V0**：`SpecDecodeWorker` 组合链路。  
2. V1 在 NPU 上主支持 `ngram` / `deepseek_mtp`，不是通用 `draft_model` 主链。  
3. `vllm-ascend` 通过 patch 覆盖 V0 的关键点：
   - `SpecDecodeWorker.create_worker`
   - `MultiStepWorker.sampler_output`

## 2. 路径分流（第一定位点）

文件：`vllm-ascend/vllm_ascend/platform.py`

- `VLLM_USE_V1=1`：
  - `worker_cls = vllm_ascend.worker.worker_v1.NPUWorker`
- 否则且有 `speculative_config`：
  - `worker_cls = vllm.spec_decode.spec_decode_worker.create_spec_worker`
  - `sd_worker_cls = vllm_ascend.worker.worker.NPUWorker`

这一步决定你后续应改 V0 还是 V1。

## 2.1 关键代码片段（代码块）

```python
# vllm-ascend/vllm_ascend/platform.py
if parallel_config and parallel_config.worker_cls == "auto":
    if envs.VLLM_USE_V1:
        parallel_config.worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"
    elif vllm_config.speculative_config:
        os.environ["ACL_OP_INIT_MODE"] = "1"
        parallel_config.worker_cls = (
            "vllm.spec_decode.spec_decode_worker.create_spec_worker")
        parallel_config.sd_worker_cls = "vllm_ascend.worker.worker.NPUWorker"
```

## 3. V0（draft_model）端到端链路

1. `create_spec_worker` 组装 scorer/proposer（`vllm/vllm/spec_decode/spec_decode_worker.py`）。
2. `SpecDecodeWorker.execute_model` 决定走 `no_spec` 还是 `_run_speculative_decoding_step`。
3. proposer 侧：`MultiStepWorker` -> 可选 `TP1DraftModelRunner` 快路。
4. scorer 侧：`MQA` 或 `batch expansion`。
5. `_verify_tokens` 用 rejection/typical sampler 验收。

NPU 对应 patch：

- `vllm-ascend/vllm_ascend/patch/worker/patch_common/patch_spec_decode_worker.py`
- `vllm-ascend/vllm_ascend/patch/worker/patch_common/patch_multi_step_worker.py`

## 4. V1 链路（只在对应方法生效）

文件：`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`

- 通过 `speculative_config.method` 选择 drafter：`ngram` / `eagle` / `deepseek_mtp`。
- 使用 `AscendRejectionSampler` 执行验收（`vllm-ascend/vllm_ascend/sample/rejection_sampler.py`）。
- 核心元数据构造点：`_calc_spec_decode_metadata`。

## 5. 常见错位与路径纠正

1. `vllm-ascend/worker/...` 是错的，实际应为 `vllm-ascend/vllm_ascend/worker/...`。  
2. `vllm-ascend/vllm_ascend/worker/spec_decode_worker.py` 不存在。V0 主体在 `vllm/vllm/spec_decode/spec_decode_worker.py`。  
3. `early_exit_threshold` 不是 `SpeculativeConfig` 内置字段，需要自定义配置接入。

## 6. 方向到代码入口（落地映射）

## 6.1 动态步长

- `vllm/vllm/spec_decode/spec_decode_worker.py`
- `vllm/vllm/spec_decode/top1_proposer.py`
- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

## 6.2 模型/表示对齐

- `vllm/vllm/spec_decode/spec_decode_worker.py`
- `vllm/vllm/spec_decode/batch_expansion.py`
- `vllm-ascend/vllm_ascend/worker/worker.py`
- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

## 6.3 CPU/NPU 切换开销

- `vllm-ascend/vllm_ascend/worker/draft_model_runner.py`
- `vllm-ascend/vllm_ascend/worker/multi_step_runner.py`
- `vllm/vllm/spec_decode/multi_step_worker.py`

## 7. 最小可复现实验入口

```powershell
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_spec_decode_worker.py -q
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_dynamic_spec_decode.py -q
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_multi_step_worker.py -q
pytest vllm-ascend/tests/long_term/spec_decode_v1/test_v1_spec_decode.py -q
```

## 8. 推荐阅读顺序

1. 本文件（代码分流与入口）  
2. `dynamic_step_adjustment.md`（动态步长）  
3. `model_alignment.md`（运行时对齐 + 表示校准）  
4. `cpu_npu_switch_overhead.md`（Host/Device 开销）
