# HS 对齐工作手册（新手可执行版）

> 适用场景：`Qwen3-8B (target) + Qwen3-0.6B (draft)`，NPU 端 vLLM / vllm-ascend `draft_model` speculative decoding。  
> 文档目标：让初学者在一个文件内完成“背景理解 -> 代码改造 -> 实验验证 -> 毕设写作落地”。  
> 版本：v2（扩展版）  
> 更新时间：2026-03-10

---

## 0. 快速开始（先看这个）

如果你时间有限，按下面顺序执行：

1. 先读第 1-4 节，明确“什么叫真 HS 对齐”。
2. 先做第 8 节路径 A（快速 POC），拿到第一轮证据。
3. 再做第 9 节路径 B（正式版），作为毕设主贡献。
4. 按第 12 节跑实验矩阵，按第 13 节门槛判定是否显著。
5. 用第 16 节模板写实验结论，避免口径错误。

---

## 1. 你在做什么（任务定义）

你当前毕业设计主题是：

- 在 NPU 端优化 vLLM speculative decoding 的效率。
- 当前重点是 `Qwen3-8B(target)` 与 `Qwen3-0.6B(draft)` 的 hidden states（简称 HS）对齐。

你要解决的问题不是“链路里有没有 HS”，而是：

- draft 是否真正消费了上一步 target 的 hidden states，
- 并且这种消费是否能稳定提升 `accept_rate / throughput`，且延迟可控。

---

## 2. 初学者必须先建立的概念

### 2.1 speculative decoding 在这里的基本角色

- `target model`：最终裁决 token 的模型（8B）。
- `draft model`：先提议 token 的模型（0.6B）。
- `accept_rate` 越高，说明 draft 提议和 target 越一致，系统效率通常越好。

### 2.2 hidden states 在这个任务里的作用

- 直觉上，HS 是 target 对当前上下文的“内部语义表示”。
- 如果 draft 能利用 target HS，通常有机会让 draft 分布更接近 target，提升 acceptance。

### 2.3 两个容易混淆的词

- `hs_available`：HS 在链路里“可用/传到了”。
- `hs_effective`：HS 被 draft 前向“真实消费并影响输出”。

**重要：** `hs_available` 上升不等于 `hs_effective` 上升。

---

## 3. 一句话技术结论（先给答案）

“真正 HS 对齐”是 **可实现（Yes）** 的。  
原因：当前代码已经具备 HS 传递链路和能力探测，缺的是 `Qwen3 draft` 的消费逻辑（runner 融合或 model 融合）。

---

## 4. 已核对事实（必须作为后续论证前提）

### 4.1 HS 链路已经能传递

- target 可返回 hidden states：
  - `vllm-ascend/vllm_ascend/worker/worker.py:102`
  - `vllm-ascend/vllm_ascend/worker/model_runner.py:1546`
- `SpecDecodeWorker` 会缓存并传递：
  - `vllm/vllm/spec_decode/spec_decode_worker.py:1017`
  - `vllm/vllm/spec_decode/spec_decode_worker.py:1251`
  - `vllm/vllm/spec_decode/spec_decode_worker.py:1274`
- request 结构支持字段：
  - `vllm/vllm/sequence.py:1390`

### 4.2 Qwen3 draft 当前默认不消费 previous_hidden_states

- Qwen3 forward 没有 `previous_hidden_states` 参数：
  - `vllm-ascend/vllm_ascend/models/qwen3.py:445`
- draft runner 只在模型支持时透传：
  - `vllm-ascend/vllm_ascend/worker/draft_model_runner.py:131`
  - `vllm-ascend/vllm_ascend/worker/draft_model_runner.py:523`

### 4.3 可参考的“已消费 HS”实现

- `deepseek_mtp` forward 明确接收 `previous_hidden_states`：
  - `vllm-ascend/vllm_ascend/models/deepseek_mtp.py:214`

---

## 5. 你现在可以直接用的资产

- 实验脚本：
  - `scripts/model_alignment_bench.sh`
  - `scripts/model_alignment_bench_v2.sh`
- 主实验文档：
  - `.github/model_alignment.md`
  - `.github/model_alignment_1plus3_experiment_plan.md`
- 历史结果数据：
  - `github/Data/*.md`

说明：本手册是“交接+执行单文件”，`model_alignment.md` 是“主实施规范”。两者内容应保持一致口径。

---

## 6. 目标拆解（你需要交付什么）

你最终要交付 4 类成果：

1. **工程成果**：路径 A/B 至少一条落地并可开关控制。
2. **实验成果**：完整指标表（吞吐、延迟、accept、hs_available、hs_effective）。
3. **分析成果**：说明提升来自什么机制（不是只报数字）。
4. **论文成果**：可写入毕设的“问题-方法-实验-结论-风险”闭环。

---

## 7. 强约束（任何时候都不能违反）

1. 默认行为必须与 baseline 一致（所有新功能默认关闭）。
2. 所有新逻辑必须可回退（单开关可禁用）。
3. 不得把“HS 可传递”写成“HS 已被消费”。
4. 结论必须同时报告吞吐、延迟、接受率，不能单指标下结论。
5. 实验至少 3 次重复，报告均值和标准差。

---

## 8. 路径 A：Runner 侧融合（快速 POC，先做）

### 8.1 目标

不改 Qwen3 模型签名，只在 `TP1DraftModelRunner` 中消费 HS，快速验证“真 HS 对齐是否有收益”。

### 8.2 推荐融合形式（低风险）

设：

- `h_prev`: 上一步 target hidden states，形状 `[B, d_t]`
- `h_cur`: draft 当前 hidden states，形状 `[B, d_d]`

步骤：

1. `h_proj = W_h(LN(h_prev))`，把 `d_t -> d_d`
2. `h_fused = h_cur + sigmoid(g) * h_proj`
3. 用 `h_fused` 走 `compute_logits`

### 8.3 最小改动点

文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

- 在 `__init__` 新增：
  - HS 融合开关读取
  - adapter 权重加载
  - gate 参数初始化
- 在 `execute_model` 中：
  - model forward 后、`compute_logits(...)` 前调用融合函数
- 新增统计：
  - `hs_input_steps`
  - `hs_effective_steps`
  - `hs_effective_ratio`

### 8.4 推荐开关（默认关闭）

- `VLLM_ASCEND_DRAFT_HS_ENABLE=0/1`
- `VLLM_ASCEND_DRAFT_HS_ADAPTER_PATH=/path/to/adapter.pt`
- `VLLM_ASCEND_DRAFT_HS_GATE_INIT=0.0`
- `VLLM_ASCEND_DRAFT_HS_FUSE_MODE=add_gate`

### 8.5 路径 A 的成功标准

- `hs_effective_ratio` 明显 > 0（建议 >= 0.95 代表几乎全步生效）
- 在 `k=4` 场景：
  - `accept_rate` 有提升（建议 >= +1.5pp）
  - 且 `iter_latency_ms_p95` 恶化 <= 3%

---

## 9. 路径 B：Model 侧融合（正式版，论文主贡献）

### 9.1 目标

把 HS 消费语义放到 `Qwen3` 模型内部，形成可解释、可维护、可复现的方法。

### 9.2 设计原则

1. `forward(..., previous_hidden_states=None, ...)` 参数为可选。
2. 默认 `None` 时行为与 baseline 完全一致。
3. HS 融合在模型早期（embedding 后）完成，避免改动过深。

### 9.3 最小改动点

文件：`vllm-ascend/vllm_ascend/models/qwen3.py`

- `CustomQwen3Model.__init__`：新增 `hs_adapter` 和 gate。
- `CustomQwen3Model.forward`：新增 `previous_hidden_states` 参数并融合。
- `CustomQwen3ForCausalLM.forward`：参数透传。

文件：`vllm-ascend/vllm_ascend/worker/draft_model_runner.py`

- 无需大改链路，复用 `_supports_previous_hidden_states` 能力探测。

### 9.4 路径 B 的成功标准

- 功能开关关时与 baseline 一致。
- 开启后在 `k=4` 场景至少满足：
  - `throughput_tps` +5% 或 `accept_rate` +2pp
  - `iter_latency_ms_p95` 不恶化超过 3%

---

## 10. 训练方案（只训小适配层，控制成本）

### 10.1 训练目标

不是重训 0.6B，而是训练“小 adapter + gate”，让 draft 更好吸收 target HS。

### 10.2 参数策略

- 冻结：draft 主干全部参数。
- 训练：`hs_adapter` + gate 参数。

### 10.3 推荐损失

- `L_kl = KL(student_logits || teacher_logits)`
- `L_ce = CE(student_logits, gt)`
- 可选 `L_h = MSE(h_adapter, h_target_proj)`
- 总损失：`L = λ1*L_kl + λ2*L_ce + λ3*L_h`

### 10.4 数据建议

训练样本字段建议包含：

- `input_ids`
- `previous_hidden_states`
- `teacher_logits`（8B 输出）
- `gt_token`

划分建议：

- 按 request 划分 train/val，防止同请求泄漏。

### 10.5 产物建议

- 输出独立 `adapter.pt`
- 推理时通过环境变量加载，支持热插拔

---

## 11. 指标与口径（统一标准）

### 11.1 主指标

- `throughput_tps`
- `iter_latency_ms_p95`
- `accept_rate`

### 11.2 对齐指标

- `hidden_states_available_ratio`
- `hs_effective_ratio`
- `accept_rate_with_hs`
- `accept_rate_no_hs`

### 11.3 关键解释指标

- `delta_accept_hs = accept_rate_with_hs - accept_rate_no_hs`

解释原则：

- `hs_available_ratio` 只能证明“链路可达”。
- `hs_effective_ratio` 才能证明“模型消费”。

---

## 12. 实验设计（建议直接照做）

### 12.1 Phase 0：基线复现

目标：确认你本机环境能稳定复现已有趋势。

对比组：

- A0：默认
- A1：ForceHS
- A2：ForceHS + 对齐（现有 logits calibration）

每组：

- `k=2/4/8`
- 至少 3 次重复

### 12.2 Phase 1：路径 A 无训练 POC

对比：

- H0：ForceHS only
- H1：路径 A（不训练）

看点：

- 是否出现非零且稳定 `hs_effective_ratio`
- `accept_rate` 是否正向变化

### 12.3 Phase 2：路径 A + 训练 adapter

对比：

- H1 vs H2

看点：

- `accept_rate` 提升幅度是否扩大
- `throughput` 是否跟随提升

### 12.4 Phase 3：路径 B 正式版

对比：

- H2 vs H4（或 H0 vs H4）

看点：

- 收益是否稳定
- 代码语义是否更清晰，便于论文论证

### 12.5 实验矩阵模板

| Case | k | HS消费实现 | Adapter训练 | throughput_tps | iter_p95_ms | accept_rate | hs_available_ratio | hs_effective_ratio |
|---|---:|---|---|---:|---:|---:|---:|---:|
| H0 | 2/4/8 | No | No |  |  |  |  |  |
| H1 | 2/4/8 | Path A | No |  |  |  |  |  |
| H2 | 2/4/8 | Path A | Yes |  |  |  |  |  |
| H3 | 2/4/8 | Path B | No |  |  |  |  |  |
| H4 | 2/4/8 | Path B | Yes |  |  |  |  |  |

---

## 13. 显著性门槛（结合你当前数据给出的建议）

根据 `github/Data` 已有趋势：`k=2` 接近上限，`k=4/8` 空间更大。  
建议把“显著”定义为：

1. `accept_rate` 绝对提升 >= 2pp；
2. `throughput_tps` 提升 >= 5%；
3. `iter_latency_ms_p95` 恶化 <= 3%。

经验预测区间：

- `k=2`: 提升可能较小（常见 `0~1.5pp`）
- `k=4`: 最容易显著（常见 `+1.5~3.5pp`）
- `k=8`: 有潜力但方差更大，建议与 adaptive-k 联动看

---

## 14. 回归测试与回退策略

### 14.1 必跑回归

```powershell
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_spec_decode_worker.py -q
pytest vllm-ascend/tests/long_term/spec_decode_v0/test_multi_step_worker.py -q
```

### 14.2 回退原则

满足任一条件立即回退：

- 出现 shape mismatch / 运行异常
- `iter_latency_ms_p95` 超门槛恶化
- 收益不稳定且方差过大

回退动作：

- 关 `VLLM_ASCEND_DRAFT_HS_ENABLE`
- 恢复 baseline 路径

---

## 15. 常见坑（新手高频）

1. 把 `hs_available` 当成“已对齐完成”。
2. 只看吞吐，不看 p95 延迟和 accept。
3. 只跑 1 次就下结论。
4. 开关默认打开，污染 baseline。
5. adapter 权重与 hidden 维度不匹配。
6. 在 tail（低并发）区间做主结论，导致误判。

---

## 16. 毕设写作建议（可直接改成正文）

建议章节结构：

1. 问题定义：Qwen3 draft 未消费 HS，导致“可用不等于有效”。
2. 方法：路径 A（POC）+ 路径 B（正式）两阶段。
3. 实验：按 H0-H4、k=2/4/8，报告主指标+对齐指标。
4. 结果：强调 `k=4` 的主要收益和显著性。
5. 风险：高 k 方差、延迟抖动、回退策略。
6. 结论：给出可部署推荐配置与未来工作。

---

## 17. 可以直接复制给 AI 助手的执行型 Prompt

```text
你现在接手一个 NPU 端 vLLM speculative decoding 优化任务，目标是实现并验证 Qwen3-8B(target) 与 Qwen3-0.6B(draft) 的“真正 HS 对齐”。

已知事实：
1) HS 在框架内可传递：spec_decode_worker 会缓存并传递 previous_hidden_states；
2) Qwen3 forward 当前不接 previous_hidden_states；
3) draft_model_runner 会做能力探测，不支持则不传；
4) deepseek_mtp 是可参考的“支持 previous_hidden_states”实现。

请输出：
A. 技术可行性判断（是否可实现真正 HS 对齐）；
B. 两条落地路线：
   - 路线A：runner 侧融合（快速POC）
   - 路线B：model 侧接口+融合（正式版）
C. 每条路线的最小改动清单（精确到文件/函数）；
D. 训练方案（仅训小适配层）与损失设计；
E. 实验矩阵和验收门槛（throughput/iter_p95/accept_rate + hs_effective_ratio）；
F. 风险与回退策略。

限制：
- 默认开关关闭；
- 必须可回退；
- 不可把 hs_available 误当作 hs_effective。
```

---

## 18. 与其他文档的关系

- 本文件：交接 + 执行手册（偏“怎么做”）。
- `.github/model_alignment.md`：主实施规范（偏“流程与验收”）。
- `.github/model_alignment_1plus3_experiment_plan.md`：1+3 联合实验（偏“组合策略”）。

建议维护规则：

- 先更新本文件，再同步主文档对应章节，避免内容分叉。
