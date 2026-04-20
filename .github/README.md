# SpecDecode-On-vLLM-Ascend

昇腾 NPU 场景下基于 vLLM-Ascend 的投机解码（Speculative Decoding）联合优化方案，包含动态步长控制器、未归一化得分对齐、多查询注意力（MQA）打分路径与快速路径（FastPath）消融等实验脚本。

> **对应论文**：湖南大学本科毕业论文《昇腾场景下基于 vLLM-Ascend 的投机解码联合优化》
>
> **分支**：`MQA_FP` → `vllm-ascend`

---

## 目录

1. [仓库结构](#1-仓库结构)
2. [环境准备](#2-环境准备)
3. [脚本基本参数说明](#3-脚本基本参数说明)
4. [各实验运行命令](#4-各实验运行命令)
5. [实验输出说明](#5-实验输出说明)
6. [常见问题排查](#6-常见问题排查)

---

## 1. 仓库结构

```
代码&&文档/
├── scripts/                          # 核心评测脚本目录
│   ├── model_alignment_bench.sh       # 基础脚本：固定 k 基线、A0/A1/A2 矩阵实验
│   ├── model_alignment_bench_v2.sh    # v2 脚本：自适应控制器、联合优化实验
│   ├── spec_patch_verify.sh           # MQA × FastPath 消融实验
│   ├── cpu_overhead_bench.sh           # CPU 开销 profiling 实验
│   ├── model_alignment_bench_repeat_avg.sh  # 重复实验汇总
│   ├── npu_process_clean_confirm.sh    # NPU 进程清理确认脚本
│   └── k_sweep_basic.sh               # 基础 k 扫描脚本（参考）
│
├── vllm/                           # vLLM v0.15 源码（对照版）
├── vllm-ascend/                      # vLLM-Ascend 源码（已 patch）
├── github/                            # GitHub 相关文档与实验数据
├── docs/                              # 实验文档
│   ├── 实验文档.md                     # 各实验的脚本调用对照表
│   └── 代码检测报告相关文档/
│       └── 代码检测文档.md             # 核心代码改动检测说明
└── 代码检测文档.md                     # 代码检测文档（根目录副本）
```

---

## 2. 环境准备

### 2.1 硬件与软件要求

| 项目 | 要求 |
|------|------|
| 硬件 | 华为昇腾 NPU（Ascend NPU） |
| 驱动 | CANN 神经网络计算架构（需正确安装并激活） |
| Python | ≥ 3.8，建议 3.10 |
| vLLM 版本 | **v0.9.1**（`VLLM_USE_V1=0`，必须） |

### 2.2 克隆仓库

```bash
# 克隆主仓库
git clone https://github.com/407407lxc/SpecDecode-On-vllm-ascend.git
cd SpecDecode-On-vllm-ascend

# 切换到 MQA_FP 分支（包含最新实验脚本）
git checkout MQA_FP
```

### 2.3 模型文件准备

本项目使用 Qwen3 系列模型，需提前下载并配置路径：

| 模型 | 用途 | 推荐路径 |
|------|------|----------|
| Qwen3-8B | 目标模型（Target） | `/mnt/ky2307909/siyuan.tong/Qwen3-8B` |
| Qwen3-0.6B | 草稿模型（Draft） | `/mnt/ky2307909/siyuan.tong/Qwen3-0.6B` |

**本地替换方式**：克隆后，在每个脚本顶部或运行时通过环境变量覆盖路径：

```bash
# 方式一：运行时环境变量（推荐）
MODEL_PATH="/path/to/your/Qwen3-8B" \
DRAFT_MODEL_PATH="/path/to/your/Qwen3-0.6B" \
bash scripts/model_alignment_bench.sh
```

### 2.4 数据集准备

基准测试使用 ShareGPT 对话数据集（V3 清洗版）：

```
/mnt/ky2307909/siyuan.tong/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

本地使用时设置：

```bash
DATASET_PATH="/path/to/ShareGPT_V3_unfiltered_cleaned_split.json"
```

### 2.5 环境变量总览

所有脚本共享以下默认环境变量，可用 `export` 或 `KEY=VALUE bash` 方式覆盖：

```bash
# 路径配置
MODEL_PATH="${MODEL_PATH:-/mnt/ky2307909/siyuan.tong/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/mnt/ky2307909/siyuan.tong/Qwen3-0.6B}"
DATASET_PATH="${DATASET_PATH:-/mnt/ky2307909/siyuan.tong/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"

# 服务配置
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-16}"

# 核心版本标志（必须为 0，对应 v0.9.1）
export VLLM_USE_V1=0

# Ascend NPU 配置
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
unset CUDA_VISIBLE_DEVICES || true
```

### 2.6 vLLM 源码环境配置

本项目使用仓库中提供的 vLLM 源码，无需从 PyPI 安装。克隆仓库后，需要将本地源码路径配置到 Python 环境变量或 `PYTHONPATH` 中：

```bash
# 方式一：通过 PYTHONPATH 优先使用本地源码（推荐）
export PYTHONPATH="/path/to/SpecDecode-On-vllm-ascend/vllm:/path/to/SpecDecode-On-vllm-ascend/vllm-ascend:$PYTHONPATH"

# 方式二：直接链接到 site-packages（需确保无外部 vLLM 安装冲突）
# 例如将 vllm-ascend 目录软链接到 Python 环境路径
```

**源码目录说明**：

| 目录 | 说明 | 用途 |
|------|------|------|
| `vllm/` | 标准 vLLM v0.15 源码（对照版） | 作为基准对比或 MQA 打分路径验证 |
| `vllm-ascend/` | vLLM-Ascend 源码（含 FastPath、Controller 等 patch） | 昇腾 NPU 场景主要运行环境 |

**验证方式**：

```bash
# 确认源码路径优先级
python -c "import vllm; print(vllm.__file__)"
python -c "import vllm_ascend; print(vllm_ascend.__file__)"
# 应输出仓库内的源码路径
```

> **注意**：请勿通过 `pip install vllm` 安装 PyPI 版本，否则会与本地源码冲突。实验脚本运行时会自动通过 `PYTHONPATH` 优先加载本地源码。

---

## 3. 脚本基本参数说明

### 3.1 `model_alignment_bench.sh`（基础脚本）

用于固定 $k$ 基线实验与 A0/A1/A2 对齐矩阵实验。

**核心参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 目标模型路径（Qwen3-8B） | 需替换 |
| `DRAFT_MODEL_PATH` | 草稿模型路径（Qwen3-0.6B） | 需替换 |
| `DATASET_PATH` | ShareGPT 数据集路径 | 需替换 |
| `HOST` / `PORT` | 服务监听地址 | `127.0.0.1` / `8000` |
| `NUM_PROMPTS` | 基准测试请求数量 | `200` |
| `REQUEST_RATE` | 基准并发请求率 | `16` |
| `RUN_A0` | 启用 A0 实验（仅草稿，无对齐） | `1` |
| `RUN_A1` | 启用 A1 实验（force_hs，开启阶段成本统计） | `1` |
| `RUN_A2` | 启用 A2 实验（logits 对齐） | `1` |
| `FIXED_K_LIST_STR` | 固定 $k$ 扫描列表，逗号分隔 | `2,4,8` |
| `A2_ALIGN_SCALE` | 对齐缩放参数 $\alpha$ | `1.0` |
| `A2_ALIGN_BIAS` | 对齐偏置参数 $b$ | `0.0` |
| `A2_ALIGN_TEMPERATURE` | 对齐温度参数 $T$ | `1.0` |
| `LOG_DIR` | 日志输出目录 | `./logs/model_alignment_时间戳` |
| `LOG_FILE_MODE` | 日志文件模式：`full`（每 case 独立）或 `two`（共用 server.log + bench.log） | `full` |
| `READY_TIMEOUT_S` | 服务启动等待超时（秒，0=无限等待） | `0` |

### 3.2 `model_alignment_bench_v2.sh`（v2 脚本）

在基础脚本之上增加了自适应控制器与联合优化实验支持，是论文第四章实验的主要驱动脚本。

**核心参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EXPERIMENT_MODE` | 实验模式：`legacy`/`basedraft`/`adaptive_only`/`align_only`/`joint`/`all` | `legacy` |
| `K_LIST` | 固定 $k$ 列表（用于 non-adaptive 模式） | 继承 `FIXED_K_LIST_STR` |
| `ADAPTIVE_INIT_K_LIST` | 自适应控制器的初始 $k$ 列表，逗号分隔 | 继承 `K_LIST` |
| `ADAPTIVE_K_MIN` | 自适应 $k$ 下限 | `2` |
| `ADAPTIVE_K_MAX` | 自适应 $k$ 上限 | `8` |
| `ADAPTIVE_ENABLE_UTILITY` | 启用增强型控制器（效用估计） | `1` |
| `LOG_ROOT` | v2 多实验的日志根目录 | `./logs/model_alignment_v2_时间戳` |

**`EXPERIMENT_MODE` 含义：**

| 模式 | 说明 |
|------|------|
| `legacy` | 保持旧脚本行为，使用 `RUN_A0/A1/A2` + `FIXED_K_LIST_STR` |
| `basedraft` | 等价于 `RUN_A0=1`，仅草稿模型基线 |
| `adaptive_only` | 自适应 / 增强型控制器实验（对应论文第 3.2 / 4.1 节） |
| `align_only` | 未归一化得分对齐实验（对应论文第 3.3 / 4.2 节） |
| `joint` | 控制器 + 对齐联合实验（对应论文第 4.2 节） |
| `all` | 运行 basedraft + align_only + adaptive_only + joint 全套 |

**v2 新增对齐门控参数（`EXPERIMENT_MODE=joint` 时自动启用）：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_ASCEND_ADAPTIVE_ALIGN_GATE_ENABLE` | 启用对齐门控 | `1`（joint 模式自动开启） |
| `VLLM_ASCEND_ADAPTIVE_ALIGN_LOW_TH` | 低接受率阈值 | `0.45` |
| `VLLM_ASCEND_ADAPTIVE_ALIGN_HIGH_TH` | 高接受率阈值 | `0.70` |
| `VLLM_ASCEND_ADAPTIVE_ALIGN_W_POS2` | 位置 2 接受率权重 | `0.45` |
| `VLLM_ASCEND_ADAPTIVE_ALIGN_W_NOWASTE` | 无浪费率权重 | `0.45` |
| `VLLM_ASCEND_ADAPTIVE_ALIGN_W_DELTA_HS` | 隐藏状态变化权重 | `0.10` |

### 3.3 `spec_patch_verify.sh`（MQA × FastPath 消融脚本）

用于 MQA 打分路径与快速路径的 A/B/C/D 四组消融实验。

**核心参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `K_LIST_STR` | 空格分隔的 $k$ 列表 | `2 4 8` |
| `SUITES_STR` | 消融 suite 列表（空格分隔） | `A_FP_ON_MQA_ON B_FP_ON_MQA_OFF C_FP_OFF_MQA_ON D_FP_OFF_MQA_OFF NO_SPEC` |
| `A_ENABLE_EWMA` | A 组启用 EWMA 平滑 | `1` |
| `A_ENABLE_MQA_LAZY` | A 组启用 MQA lazy 模式 | `1` |
| `NON_A_ENABLE_EWMA` | 非 A 组启用 EWMA | `0` |
| `NON_A_ENABLE_MQA_LAZY` | 非 A 组启用 MQA lazy | `0` |
| `VLLM_ASCEND_ADAPTIVE_EWMA_BETA` | EWMA 平滑 beta 系数 | `0.2` |
| `LOG_ROOT` | 日志根目录 | 自动生成 |

**消融 suite 含义：**

| Suite | FastPath | MQA 打分路径 | 说明 |
|-------|----------|--------------|------|
| `A_FP_ON_MQA_ON` | 开 | MQA（多查询注意力） | 全部联合最优 |
| `B_FP_ON_MQA_OFF` | 开 | 批扩展（BatchExpansion） | 仅 FastPath |
| `C_FP_OFF_MQA_ON` | 关 | MQA | 仅 MQA |
| `D_FP_OFF_MQA_OFF` | 关 | 批扩展 | 基线（legacy） |
| `NO_SPEC` | — | — | 无投机解码基线 |

**也可单独运行某个 suite：**

```bash
SELECTED_SUITES="A_FP_ON_MQA_ON" bash scripts/spec_patch_verify.sh
```

### 3.4 `cpu_overhead_bench.sh`（CPU 开销 profiling 脚本）

用于分析提案阶段 CPU 预处理开销与 FastPath 命中 / 回退细分原因。

**核心参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `RUN_C0` | 启用 C0 case（logprobs 关，$k=1$） | `1` |
| `RUN_C1` | 启用 C1 case（logprobs 关，$k=8$） | `1` |
| `RUN_C2` | 启用 C2 case（logprobs 开，$k=8$） | `1` |
| `C0_K` / `C1_K` / `C2_K` | 各 case 的固定 $k$ 值 | `1` / `8` / `8` |

**Case 含义：**

| Case | logprobs | $k$ 值 | 主要验证内容 |
|------|----------|--------|------------|
| C0 | 关闭 | 1 | 无额外开销的基线 CPU 预处理 |
| C1 | 关闭 | 8 | 中等步长下的 CPU 预处理 |
| C2 | 开启 | 8 | logprobs 开启后的 CPU 开销变化 |

---

## 4. 各实验运行命令

> **前提**：请先确认已将 `MODEL_PATH`、`DRAFT_MODEL_PATH`、`DATASET_PATH` 替换为本地实际路径。

### 4.1 固定 $k$ 基线实验

对应论文：**第 3.1 节「数据观察与分析」**

建立 $k=2,4,8$ 下的吞吐、接受率与阶段耗时基线。

```bash
cd 代码&&文档/scripts

MODEL_PATH="/path/to/Qwen3-8B" \
DRAFT_MODEL_PATH="/path/to/Qwen3-0.6B" \
DATASET_PATH="/path/to/ShareGPT_V3_unfiltered_cleaned_split.json" \
LOG_DIR="./logs/fixed_k_baseline_$(date +%Y%m%d_%H%M%S)" \
LOG_FILE_MODE=full \
bash model_alignment_bench.sh
```

**仅运行 A0 基线（快速建立接受率基线）：**

```bash
MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
RUN_A0=1 RUN_A1=0 RUN_A2=0 \
FIXED_K_LIST_STR="2,4,8" \
LOG_DIR="./logs/fixed_k_A0_baseline" \
LOG_FILE_MODE=full \
bash model_alignment_bench.sh
```

**仅运行固定 $k=4$（用于后续与控制器对比）：**

```bash
MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
RUN_A0=1 RUN_A1=0 RUN_A2=0 \
FIXED_K_LIST_STR="4" \
LOG_DIR="./logs/fixed_k4_baseline" \
LOG_FILE_MODE=full \
bash model_alignment_bench.sh
```

---

### 4.2 控制器实验

对应论文：**第 3.2 节「动态投机步长控制器设计」**、**第 4.1 节「控制器结果分析」**

**自适应控制器实验（init\_k 扫描 2,4,8）：**

```bash
cd 代码&&文档/scripts

MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
EXPERIMENT_MODE=adaptive_only \
ADAPTIVE_INIT_K_LIST="2,4,8" \
ADAPTIVE_K_MIN=2 \
ADAPTIVE_K_MAX=8 \
ADAPTIVE_ENABLE_UTILITY=1 \
LOG_ROOT="./logs/controller_adaptive_$(date +%Y%m%d_%H%M%S)" \
LOG_FILE_MODE=full \
bash model_alignment_bench_v2.sh
```

**增强型控制器实验（`adaptive_pro`，即 `ADAPTIVE_ENABLE_UTILITY=1` 的自适应模式）：**

增强型控制器由代码内部实现，v2 脚本已覆盖上述参数组合即为增强型模式。

**关闭 EWMA 平滑（使用原始瞬时值决策）：**

```bash
export VLLM_ASCEND_ADAPTIVE_EWMA_ENABLE=0
bash model_alignment_bench_v2.sh
```

---

### 4.3 未归一化得分对齐实验

对应论文：**第 3.3 节「草稿模型与目标模型的未归一化得分对齐」**、**第 4.2 节「未归一化得分对齐结果分析」**

**通过基础脚本运行（固定 $k$ 扫描）：**

```bash
MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
RUN_A0=0 RUN_A1=0 RUN_A2=1 \
FIXED_K_LIST_STR="2,4,8" \
LOG_DIR="./logs/logits_align_fixed_k_$(date +%Y%m%d_%H%M%S)" \
LOG_FILE_MODE=full \
bash model_alignment_bench.sh
```

**通过 v2 脚本运行（支持更多对齐参数）：**

```bash
MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
EXPERIMENT_MODE=align_only \
K_LIST="2,4,8" \
LOG_ROOT="./logs/logits_align_v2_$(date +%Y%m%d_%H%M%S)" \
LOG_FILE_MODE=full \
bash model_alignment_bench_v2.sh
```

**调整对齐温度（$T \neq 1.0$）：**

```bash
A2_ALIGN_TEMPERATURE=0.8 bash model_alignment_bench.sh
```

**调整对齐缩放与偏置：**

```bash
A2_ALIGN_SCALE=1.2 A2_ALIGN_BIAS=-0.1 bash model_alignment_bench.sh
```

---

### 4.4 动态控制器 + 对齐联合实验

对应论文：**第 4.2 节「未归一化得分对齐结果分析」**（表 4.2 / 表 4.3 中"动态 $k$ + 对齐"行）

```bash
MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
EXPERIMENT_MODE=joint \
ADAPTIVE_INIT_K_LIST="2,4,8" \
ADAPTIVE_K_MIN=2 \
ADAPTIVE_K_MAX=8 \
ADAPTIVE_ENABLE_UTILITY=1 \
LOG_ROOT="./logs/joint_adaptive_align_$(date +%Y%m%d_%H%M%S)" \
LOG_FILE_MODE=full \
bash model_alignment_bench_v2.sh
```

> `joint` 模式同时开启 `adaptive_only`（RUN_A1=1）和 `align_only`（RUN_A2=1），并自动启用对齐门控（`VLLM_ASCEND_ADAPTIVE_ALIGN_GATE_ENABLE=1`）。

---

### 4.5 MQA 打分路径与快速路径消融实验

对应论文：**第 4.3 节「多查询注意力打分路径（MQA）与快速路径（FastPath）消融分析」**

**全套四组消融（A/B/C/D），$k=2,4,8$：**

```bash
cd 代码&&文档/scripts

MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
NUM_PROMPTS=200 \
REQUEST_RATE=16 \
LOG_DIR="./logs/mqa_fp_abcd_$(date +%Y%m%d_%H%M%S)" \
bash spec_patch_verify.sh
```

**仅运行 A 组（MQA + FastPath 均开启）：**

```bash
LOG_DIR="./logs/mqa_fp_A_only" \
SELECTED_SUITES="A_FP_ON_MQA_ON" \
bash spec_patch_verify.sh
```

**仅运行 C 组（MQA 开启，FastPath 关闭）：**

```bash
LOG_DIR="./logs/mqa_fp_C_only" \
SELECTED_SUITES="C_FP_OFF_MQA_ON" \
bash spec_patch_verify.sh
```

**开启 EWMA 辅助实验：**

```bash
export VLLM_ASCEND_ADAPTIVE_EWMA_ENABLE=1
bash spec_patch_verify.sh
```

**开启 MQA lazy 模式：**

```bash
export VLLM_ASCEND_SPEC_MQA_LAZY_ENABLE=1
bash spec_patch_verify.sh
```

---

### 4.6 CPU 开销 profiling 实验

对应论文：**第 3.4 / 3.5 节**

```bash
cd 代码&&文档/scripts

MODEL_PATH="..." DRAFT_MODEL_PATH="..." DATASET_PATH="..." \
LOG_DIR="./logs/cpu_overhead_$(date +%Y%m%d_%H%M%S)" \
bash cpu_overhead_bench.sh
```

---

### 4.7 重复实验与结果汇总

用于对某一配置进行多次重复实验以评估稳定性：

```bash
cd 代码&&文档/scripts

REPEAT_TIMES=10 \
INNER_SCRIPT="/path/to/model_alignment_bench_minimal.sh" \
ROOT_LOG_DIR="./logs/repeat_avg_$(date +%Y%m%d_%H%M%S)" \
bash model_alignment_bench_repeat_avg.sh
```

输出：
- `aggregate_summary.csv`：各配置下的均值 ± 标准差
- `aggregate_raw_rows.csv`：全部原始行的拼接文件

---

## 5. 实验输出说明

### 5.1 `model_alignment_bench.sh` / `model_alignment_bench_v2.sh` 输出

| 输出文件 | 内容 |
|----------|------|
| `summary.csv` | 各 case 的吞吐、TTFT、TPOT、ITL、接受率、系统效率等汇总指标 |
| `server_<tag>.log` | 各 case 的 vLLM 服务端日志（`LOG_FILE_MODE=full` 时） |
| `bench_<tag>.log` | 各 case 的 benchmark 压测结果 |
| `key_<tag>.log` | 关键行抽取（阶段均值、接受率、AdaptiveK 统计等） |
| `merged_<tag>.log` | server + bench 合并日志 |
| `result_<tag>.log` | case 级汇总结果（含 benchmark 摘要表格） |

**控制器实验额外输出：**

| 字段 | 含义 |
|------|------|
| `adaptive_switch_per_min` | 每分钟运行点切换次数 |
| `adaptive_high_k_occ` | 高 $k$ 出现频率 |
| `adaptive_hist` | $k$ 值直方图统计 |

**对齐实验额外输出（来自服务端 `ALIGN P1/P2` 日志行）：**

| 字段 | 含义 |
|------|------|
| `align_hs_ratio` | 隐藏状态有效性比例 |
| `align_accept` | 标准接受率 |
| `align_accept_with_hs` / `align_accept_no_hs` | 有/无隐藏状态条件下的接受率 |
| `align_tps` | 对齐后的吞吐估计 |
| `align_p95_ms` | 对齐后的 P95 迭代耗时 |

### 5.2 `spec_patch_verify.sh` 输出

| 字段 | 含义 |
|------|------|
| `mqa_mode` | 当前打分路径类型（MQA / BATCH_EXPANSION） |
| `fastpath_hit_rate` | FastPath 命中率 |
| `scoring_ms` | 打分阶段总耗时（均值） |
| `verify_ms` | 验证阶段耗时（均值） |
| `stage_scoring_share` | 打分阶段占总耗时比例 |
| `stage_p95_iter_ms` | 单轮迭代 P95 耗时 |
| `mqa_total_ms` / `mqa_execute_ms` | MQA 打分器总耗时 / 执行耗时 |
| `base_worker_avg_total_ms` / `base_p95_spmd_ms` | BaseWorker 阶段耗时 |
| `base_runner_avg_total_ms` / `base_p95_total_ms` | BaseRunner 阶段耗时 |

### 5.3 `cpu_overhead_bench.sh` 输出

| 字段 | 含义 |
|------|------|
| `draft_acceptance_rate` / `system_efficiency` | 接受率与系统效率 |
| `ms_hit_rate` | FastPath 命中率 |
| `ms_fail_prompt` / `ms_fail_backend` / `ms_fail_lora` / `ms_fail_adapter` | FastPath 失败原因细分 |

---

## 6. 常见问题排查

### Q1：服务启动后 benchmark 超时无响应

**原因**：NPU 进程残留或 `READY_TIMEOUT_S` 不够。

**解决方案**：

```bash
# 清理残留进程
bash npu_process_clean_confirm.sh

# 确认服务就绪后再运行
READY_TIMEOUT_S=600 bash model_alignment_bench.sh
```

### Q2：MQA 路径始终回退到 BatchExpansion

**原因**：`VLLM_ASCEND_SPEC_MQA_BACKENDS` 未包含当前后端名称，或后端不支持 MQA。

**解决方案**：查看 `spec_patch_verify.sh` 输出的 `mqa_mode` 字段确认路径类型；检查后端配置是否正确注册 MQA 支持。

### Q3：FastPath 命中率过低

**解决方案**：查看 `server.log` 中 `DraftGPUFastPath` 相关的失败计数器：

```
fail_prompt   → 请求条件不满足
fail_backend  → 后端不支持
fail_lora     → LoRA 模块冲突
fail_adapter  → Adapter 模块冲突
```

根据具体失败原因调整请求配置或后端设置。

### Q4：对齐实验接受率反而下降

**解决方案**：尝试降低对齐温度（$T=0.8$）或调整对齐缩放：

```bash
A2_ALIGN_TEMPERATURE=0.8 bash model_alignment_bench.sh
# 或
A2_ALIGN_SCALE=1.2 A2_ALIGN_BIAS=-0.1 bash model_alignment_bench.sh
```

logits 对齐在深位置（$k=8$）收益有限，主要价值在于 TTFT 改善。

### Q5：实验重复运行时报端口占用

**解决方案**：在每次运行前清理残留进程：

```bash
pkill -TERM -f "vllm serve|api_server.py|engine.py" 2>/dev/null || true
sleep 2
bash model_alignment_bench.sh
```

或修改 `PORT` 环境变量使用不同端口：

```bash
PORT=8001 bash model_alignment_bench.sh
```

---

## 附录：论文表格与脚本对应关系

| 论文表格 | 对应实验 | 调用命令 |
|----------|----------|----------|
| 表 3.1 / 表 3.2（固定 $k$ 基线） | `model_alignment_bench.sh` | `RUN_A0=1 RUN_A1=0 RUN_A2=0 FIXED_K_LIST_STR="2,4,8"` |
| 表 4.1（控制器对比） | `model_alignment_bench_v2.sh` | `EXPERIMENT_MODE=adaptive_only ADAPTIVE_INIT_K_LIST="2,4,8"` |
| 表 4.2 / 表 4.3（对齐结果） | `model_alignment_bench.sh` 或 `model_alignment_bench_v2.sh` | `RUN_A2=1` 或 `EXPERIMENT_MODE=align_only` |
| 表 4.4 / 表 4.5（MQA × FastPath） | `spec_patch_verify.sh` | 默认全量运行（4 组 A/B/C/D） |
| 表 4.6 / 4.7（联合 vs 单点） | `model_alignment_bench_v2.sh` | `EXPERIMENT_MODE=joint` |

---

*本文档对应毕设论文终稿（v1.0），所有脚本基于 vLLM-Ascend v0.9.1（`VLLM_USE_V1=0`）编写。*
