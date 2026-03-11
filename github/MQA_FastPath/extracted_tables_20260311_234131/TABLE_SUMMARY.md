# MQA FastPath Data Extraction Summary

Generated at: 2026-03-11 23:42:08
Output directory: C:\Users\407407\Desktop\vllm\github\MQA_FastPath\extracted_tables_20260311_234131

## File Overview
| data_group | file_count | log_count | csv_count | total_size_bytes |
|---|---:|---:|---:|---:|
| A&B | 27 | 26 | 1 | 4233477 |
| C&D | 27 | 26 | 1 | 4157914 |
| no_spec | 6 | 5 | 1 | 643248 |

## Summary Rows (from summary.csv)
| data_group | suite | k | status | throughput_tps | mean_tpot_ms | mean_itl_ms | mqa_mode | supports_gpu_multi_step | scoring_time_ms | verification_time_ms |
|---|---|---:|---|---:|---:|---:|---|---|---:|---:|
| A&B | A_FP_ON_MQA_ON | 2 | OK | 513.78 | 307.19 | 311.45 | MQA | True | 57.61 | 5.15 |
| A&B | A_FP_ON_MQA_ON | 4 | OK | 519.02 | 326.97 | 340.11 | MQA | True | 60.61 | 5.35 |
| A&B | A_FP_ON_MQA_ON | 8 | OK | 542.98 | 302.78 | 343.23 | MQA | True | 81.52 | 5.22 |
| A&B | B_FP_ON_MQA_OFF | 2 | OK | 327.65 | 303.48 | 314.90 | BATCH_EXPANSION | True | 84.77 | 7.18 |
| A&B | B_FP_ON_MQA_OFF | 4 | OK | 314.75 | 352.96 | 332.24 | BATCH_EXPANSION | True | 56.04 | 3.56 |
| A&B | B_FP_ON_MQA_OFF | 8 | OK | 328.03 | 316.49 | 319.65 | BATCH_EXPANSION | True | 51.63 | 4.48 |
| A&B | C_FP_OFF_MQA_ON | 2 | SERVER_FAIL |  |  |  | MQA |  |  |  |
| C&D | C_FP_OFF_MQA_ON | 2 | OK | 473.06 | 316.05 | 334.03 | MQA | False. | 54.69 | 3.57 |
| C&D | C_FP_OFF_MQA_ON | 4 | OK | 493.59 | 312.44 | 358.60 | MQA | False. | 84.44 | 5.14 |
| C&D | C_FP_OFF_MQA_ON | 8 | OK | 524.13 | 330.46 | 381.50 | MQA | False. | 58.80 | 4.96 |
| C&D | D_FP_OFF_MQA_OFF | 2 | OK | 400.65 | 263.80 | 375.08 | BATCH_EXPANSION | False. | 60.01 | 5.35 |
| C&D | D_FP_OFF_MQA_OFF | 4 | OK | 403.54 | 252.10 | 396.68 | BATCH_EXPANSION | False. | 59.73 | 5.15 |
| C&D | D_FP_OFF_MQA_OFF | 8 | OK | 374.72 | 259.96 | 452.66 | BATCH_EXPANSION | False. | 54.21 | 4.84 |
| no_spec | NO_SPEC | 0 | OK | 706.73 | 139.53 | 92.72 | UNKNOWN |  |  |  |

## Output Tables
| file | description |
|---|---|
| file_inventory.csv | All files with metadata, line count and SHA256 |
| summary_combined.csv | Merged rows from all summary.csv files |
| master_run_table.csv | Run-level merged table (summary + bench + server issue stats) |
| meta_probe_kv.csv | Parsed key/value pairs from META and PROBE logs |
| bench_metrics.csv | Parsed benchmark result metrics from bench logs |
| log_issue_stats.csv | WARNING/ERROR/Traceback/SERVER_FAIL counts per log file |
| log_key_values.csv | All key=value pairs extracted from all log lines |
| log_last_values.csv | Last observed value for each key in each log file |
| log_numeric_stats.csv | Numeric stats (count/min/max/avg/last) per key and file |
| all_log_lines.csv | Full raw log lines with source path and line number |
