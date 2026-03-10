# .github 文档索引（按方向）

## 1. 代码总览

- `spec_decode_npu_flow.md`  
  作用：确认 V0/V1 分流、真实入口文件、常见路径误判。

## 2. 动态步长方向

- `dynamic_step_adjustment.md`  
  作用：实现 `Adaptive-K` 与可选 early-stop，附测试与验收标准。

## 3. 对齐方向

- `model_alignment.md`  
  作用：运行时状态对齐（必做）+ 表示校准（可选）的统一实现指南。
- `model_alignment_1plus3_experiment_plan.md`  
  作用：`logit` 分布对齐（方向1）与对齐感知 `adaptive-k`（方向3）的联合实验设计（详细版）。

## 4. Host/Device 开销方向

- `cpu_npu_switch_overhead.md`  
  作用：NPU↔CPU 切换开销分解、快路命中率优化与验收方案。

## 5. 毕设执行主线

- `thesis_idea_recommended_controller.md`  
  作用：12 周任务拆解与论文贡献组织。

---

说明：

1. 本目录已合并旧的重复文档，采用“一个方向一个主文档”的结构。  
2. 所有实验统一使用三主指标：`throughput_tps`、`iter_latency_ms_p50/p95`、`accept_rate`。

## 6. 实验脚本（新增）

- `scripts/model_alignment_bench_v2.sh`：在原 A0/A1/A2 流程上，新增 `align_mode/temperature/vocab_bias` 开关。
- `scripts/cpu_overhead_bench.sh`：一键执行 C0/C1/C2，并汇总 `DraftGPUFastPath` 命中与失败原因。
