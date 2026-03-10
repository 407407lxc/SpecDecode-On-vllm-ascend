# qwen3-8B + qwen3-0.6B (speculative_tokens=2) Benchmark Report

## Run Profile
- target_model: qwen3-8B
- draft_model: qwen3-0.6B
- num_speculative_tokens: 2
- traffic_request_rate: 16.0
- burstiness_factor: 1.0 (Poisson process)
- benchmark_completion: 200/200 requests
- log_window: 13:08:43 -> 13:10:38

## Final Serving Benchmark Result
| Metric | Value |
|---|---:|
| Successful requests | 200 |
| Benchmark duration (s) | 107.77 |
| Total input tokens | 43560 |
| Total generated tokens | 44697 |
| Request throughput (req/s) | 1.86 |
| Output token throughput (tok/s) | 414.74 |
| Total token throughput (tok/s) | 818.92 |
| Mean TTFT (ms) | 254.25 |
| Median TTFT (ms) | 225.28 |
| P99 TTFT (ms) | 559.71 |
| Mean TPOT (ms) | 253.54 |
| Median TPOT (ms) | 183.14 |
| P99 TPOT (ms) | 789.67 |
| Mean ITL (ms) | 357.89 |
| Median ITL (ms) | 303.28 |
| P99 ITL (ms) | 1549.36 |

## Key Runtime Timeline (metrics.py:417)
| Time | Prompt throughput (tok/s) | Generation throughput (tok/s) | Running reqs | GPU KV cache usage |
|---|---:|---:|---:|---:|
| 13:08:43 | 2955.1 | 110.2 | 185 | 36.4% |
| 13:08:48 | 0.0 | 411.8 | 143 | 26.8% |
| 13:08:58 | 0.0 | 384.9 | 118 | 27.3% |
| 13:09:08 | 0.0 | 332.8 | 91 | 25.6% |
| 13:09:24 | 0.0 | 240.0 | 58 | 23.1% |
| 13:09:45 | 0.0 | 98.6 | 23 | 12.5% |
| 13:10:00 | 0.0 | 33.8 | 5 | 2.6% |
| 13:10:16 | 0.0 | 14.4 | 1 | 0.5% |
| 13:10:28 | 0.0 | 1.3 | 0 | 0.0% |
| 13:10:38 | 0.0 | 0.0 | 0 | 0.0% |

## Key Speculative Timeline (metrics.py:439)
| Time | Draft acceptance rate | System efficiency | Accepted tokens | Draft tokens | Emitted tokens |
|---|---:|---:|---:|---:|---:|
| 13:08:43 | 0.639 | 0.697 | 610 | 954 | 998 |
| 13:08:58 | 0.709 | 0.749 | 9298 | 13116 | 14740 |
| 13:09:24 | 0.758 | 0.788 | 20017 | 26422 | 31226 |
| 13:09:50 | 0.775 | 0.804 | 27030 | 34864 | 42041 |
| 13:10:00 | 0.777 | 0.805 | 28200 | 36312 | 43850 |
| 13:10:16 | 0.776 | 0.804 | 28732 | 37036 | 44688 |

## Key Stage-Time Timeline (spec_decode_worker.py:1115)
| Time | avg_time_per_proposal_tok_ms | scoring_time_ms | verification_time_ms |
|---|---:|---:|---:|
| 13:08:47 | 79.53 | 116.91 | 13.72 |
| 13:09:03 | 72.73 | 105.19 | 10.01 |
| 13:09:25 | 63.37 | 87.97 | 7.04 |
| 13:09:46 | 54.76 | 96.51 | 6.13 |
| 13:10:02 | 65.54 | 56.88 | 5.25 |
| 13:10:16 | 66.17 | 57.01 | 5.21 |


## P1P2 Timeline (spec_decode_worker.py:857, p12_log_interval=10)
| Time | k | step_accept | waste_ratio |
|---|---:|---|---:|
| 13:54:50 | 2 | [1.0, 0.6368] | 0.1816 |
| 13:54:54 | 2 | [1.0, 0.6465] | 0.1768 |
| 13:54:57 | 2 | [1.0, 0.6550] | 0.1725 |
| 13:55:00 | 2 | [1.0, 0.6628] | 0.1686 |
| 13:55:04 | 2 | [1.0, 0.6713] | 0.1644 |
| 13:55:07 | 2 | [1.0, 0.6802] | 0.1599 |
| 13:55:10 | 2 | [1.0, 0.6892] | 0.1554 |
| 13:55:13 | 2 | [1.0, 0.6973] | 0.1513 |
| 13:55:16 | 2 | [1.0, 0.7056] | 0.1472 |
| 13:55:19 | 2 | [1.0, 0.7136] | 0.1432 |
| 13:55:22 | 2 | [1.0, 0.7213] | 0.1393 |
| 13:55:25 | 2 | [1.0, 0.7286] | 0.1357 |
| 13:55:28 | 2 | [1.0, 0.7339] | 0.1331 |
| 13:55:30 | 2 | [1.0, 0.7398] | 0.1301 |
| 13:55:33 | 2 | [1.0, 0.7438] | 0.1281 |
| 13:55:36 | 2 | [1.0, 0.7493] | 0.1254 |
| 13:55:39 | 2 | [1.0, 0.7533] | 0.1233 |
| 13:55:42 | 2 | [1.0, 0.7568] | 0.1216 |
| 13:55:45 | 2 | [1.0, 0.7583] | 0.1209 |
| 13:55:48 | 2 | [1.0, 0.7586] | 0.1207 |
| 13:55:51 | 2 | [1.0, 0.7604] | 0.1198 |
| 13:55:53 | 2 | [1.0, 0.7622] | 0.1189 |
| 13:55:56 | 2 | [1.0, 0.7634] | 0.1183 |
| 13:55:58 | 2 | [1.0, 0.7653] | 0.1174 |
| 13:56:01 | 2 | [1.0, 0.7685] | 0.1157 |
| 13:56:03 | 2 | [1.0, 0.7686] | 0.1157 |
| 13:56:05 | 2 | [1.0, 0.7715] | 0.1142 |
| 13:56:07 | 2 | [1.0, 0.7725] | 0.1137 |
| 13:56:09 | 2 | [1.0, 0.7648] | 0.1176 |
| 13:56:11 | 2 | [1.0, 0.7574] | 0.1213 |
| 13:56:13 | 2 | [1.0, 0.7538] | 0.1231 |
| 13:56:15 | 2 | [1.0, 0.7518] | 0.1241 |
| 13:56:17 | 2 | [1.0, 0.7459] | 0.1270 |
| 13:56:19 | 2 | [1.0, 0.7467] | 0.1267 |
| 13:56:21 | 2 | [1.0, 0.7492] | 0.1254 |

## P1P2 Trend Notes
- `k=2` throughout the run.
- Step-2 acceptance rises from `0.6368` to a peak of `0.7725` (13:56:07), then declines in the tail to the `0.746-0.749` range.
- `waste_ratio` decreases from `0.1816` to a minimum `0.1137`, then rebounds to around `0.125` near the end.
## Consolidated Trend Notes
- Draft acceptance rate increases from 0.639 to ~0.776-0.777, then stabilizes.
- System efficiency improves from 0.697 to ~0.804-0.805.
- Early and sustained generation throughput is higher than other settings, with a peak of 411.8 tok/s.
- Runtime queue and GPU KV cache usage decline smoothly to zero at tail.
- Stage verification latency drops from 13.72 ms to ~5 ms range.

## Source
- Source data: user-provided spec2 runtime logs and benchmark summary in this conversation.