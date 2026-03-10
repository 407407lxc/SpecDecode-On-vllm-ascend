# 推荐毕设主线：NPU 端 Spec-Decoding 自适应控制器（整合版）

> 建议题目：**《面向 NPU 推理服务的 Speculative Decoding 自适应控制与系统优化》**

## 1. 目录使用方式

当前 `.github` 已整理为“一个方向一个主文档”：

1. `spec_decode_npu_flow.md`：代码总览与真实入口。  
2. `dynamic_step_adjustment.md`：动态步长（主贡献）。  
3. `model_alignment.md`：运行时对齐 + 表示校准。  
4. `cpu_npu_switch_overhead.md`：Host/Device 开销优化。

## 2. 统一主指标（所有实验都必须报）

1. `throughput_tps = accepted_tokens_total / wall_time_seconds`  
2. `iter_latency_ms_p50/p95 = spec step 延迟分位数`  
3. `accept_rate = accepted_tokens_total / proposed_tokens_total`

说明：每个 baseline、每个 P1/P2、每个最终优化结果都用同一口径，避免“不可比”。

## 3. 每个思路的 Baseline、问题检验实验与必报指标

| 思路 | Baseline | 检验问题实验 | 必报指标 |
|---|---|---|---|
| 动态步长 | `B0-FixedK2`, `B1-FixedK4`, `B2-FixedK8`, `B3-DisableByBatch` | `P1` 分步接受率衰减, `P2` 无效草稿占比 | `throughput_tps`, `iter_latency_ms_p95`, `accept_rate` |
| 模型对齐 | `A0-Default`, `A1-ForceHS`, `A2-ForceHS+Calib` | `P1` hidden states 可用率, `P2` 有/无 hidden states 接受率差 | `throughput_tps`, `iter_latency_ms_p95`, `accept_rate`, `hidden_states_available_ratio` |
| CPU/NPU 开销 | `C0-K1`, `C1-K8-LogprobsOn`, `C2-K8-LogprobsOff` | `P1` 快路命中率分解, `P2` D2H 与 pythonization 成本 | `throughput_tps`, `iter_latency_ms_p95`, `accept_rate`, `gpu_multi_step_hit_rate` |

## 4. 推荐贡献结构（三章）

## 4.1 贡献 A（必做）：动态步长控制

- 目标：在线调整 `k`，提升吞吐并控制延迟。
- 交付：`Adaptive-K` 控制器 + 消融实验。

## 4.2 贡献 B（增强）：对齐感知降级

- 目标：对齐质量下降时自动降 `k` 或临时 no-spec。
- 交付：对齐可观测指标 + 联动策略。

## 4.3 贡献 C（系统）：CPU/NPU 切换治理

- 目标：提高设备多步快路命中率，减少 pythonization 开销。
- 交付：瓶颈分解图 + 优化前后对比。

## 5. 12 周执行计划

1. 第 1-2 周：基线复现与指标脚手架。  
2. 第 3-5 周：完成 A（动态步长）。  
3. 第 6-8 周：完成 B（对齐联动）。  
4. 第 9-10 周：完成 C（切换开销治理）。  
5. 第 11-12 周：消融、作图、论文定稿。

## 6. 最小毕业标准（建议）

1. 至少 2 个 workload 上，相比固定 `k` 有显著收益。  
2. 提供 A / A+B / A+B+C 的完整消融。  
3. 关键改动可通过现有 spec_decode 长测入口。  
4. 关键图表都包含 `throughput / latency / acceptance` 三主指标。

## 7. 风险控制

- 若 B/C 收益不稳定，保留 A 也可形成完整毕业主线。  
- 先保证“可运行+可测量+可复现”，再追求复杂算法。
