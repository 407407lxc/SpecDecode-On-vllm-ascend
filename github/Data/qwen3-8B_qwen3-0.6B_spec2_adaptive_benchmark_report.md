# qwen3-8B + qwen3-0.6B (speculative_tokens=2, adaptive-k) Benchmark Report

## Run Profile
- mode: adaptive k optimization
- target_model: qwen3-8B
- draft_model: qwen3-0.6B
- base_num_speculative_tokens: 2
- benchmark_completion: 200/200 requests
- log_window: 16:33:21 -> 16:35:07

## Final Serving Benchmark Result (latest)
| Metric | Value |
|---|---:|
| Successful requests | 200 |
| Benchmark duration (s) | 118.57 |
| Total input tokens | 43560 |
| Total generated tokens | 44697 |
| Request throughput (req/s) | 1.69 |
| Output token throughput (tok/s) | 376.97 |
| Total token throughput (tok/s) | 744.35 |
| Mean TTFT (ms) | 270.51 |
| Median TTFT (ms) | 251.73 |
| P99 TTFT (ms) | 549.86 |
| Mean TPOT (ms) | 304.55 |
| Median TPOT (ms) | 198.53 |
| P99 TPOT (ms) | 1377.33 |
| Mean ITL (ms) | 451.97 |
| Median ITL (ms) | 397.50 |
| P99 ITL (ms) | 2095.75 |

## Runtime Throughput Timeline (metrics.py:417)
| Time | Prompt throughput (tok/s) | Generation throughput (tok/s) | Running reqs | GPU KV cache usage |
|---|---:|---:|---:|---:|
| 16:33:22 | 2238.4 | 174.7 | 183 | 36.8% |
| 16:33:28 | 0.0 | 382.8 | 141 | 26.6% |
| 16:33:33 | 0.0 | 330.1 | 129 | 27.0% |
| 16:33:38 | 0.0 | 303.6 | 118 | 28.2% |
| 16:33:49 | 0.0 | 244.9 | 93 | 26.1% |
| 16:34:09 | 0.0 | 165.4 | 53 | 22.3% |
| 16:34:20 | 0.0 | 118.8 | 38 | 18.1% |
| 16:34:30 | 0.0 | 96.3 | 24 | 12.6% |
| 16:34:45 | 0.0 | 47.2 | 7 | 4.1% |
| 16:34:56 | 0.0 | 24.4 | 5 | 3.6% |
| 16:35:06 | 0.0 | 11.8 | 1 | 0.6% |

## Speculative Metrics Timeline (metrics.py:439)
| Time | Draft acceptance rate | System efficiency | Accepted tokens | Draft tokens | Emitted tokens |
|---|---:|---:|---:|---:|---:|
| 16:33:22 | 0.630 | 0.680 | 585 | 928 | 947 |
| 16:33:28 | 0.683 | 0.720 | 3814 | 5588 | 6039 |
| 16:33:33 | 0.695 | 0.684 | 7086 | 10198 | 10466 |
| 16:33:38 | 0.716 | 0.682 | 10601 | 14813 | 15156 |
| 16:33:49 | 0.730 | 0.677 | 13807 | 18911 | 19205 |
| 16:33:54 | 0.740 | 0.673 | 17252 | 23308 | 23532 |
| 16:34:04 | 0.754 | 0.662 | 22976 | 30487 | 30296 |
| 16:34:15 | 0.762 | 0.664 | 27580 | 36217 | 36059 |
| 16:34:25 | 0.763 | 0.669 | 30011 | 39348 | 39470 |
| 16:34:40 | 0.763 | 0.677 | 32231 | 42250 | 42925 |
| 16:34:56 | 0.762 | 0.680 | 33136 | 43464 | 44335 |
| 16:35:01 | 0.762 | 0.681 | 33322 | 43716 | 44632 |
| 16:35:06 | 0.762 | 0.681 | 33420 | 43860 | 44788 |

## Stage Time Timeline (spec_decode_worker.py:1277)
| Time | avg_time_per_proposal_tok_ms | scoring_time_ms | verification_time_ms |
|---|---:|---:|---:|
| 16:33:21 | 85.64 | 170.24 | 17.71 |
| 16:33:27 | 79.65 | 120.76 | 14.34 |
| 16:33:33 | 82.22 | 108.65 | 11.68 |
| 16:33:38 | 73.47 | 128.93 | 17.92 |
| 16:33:44 | 73.02 | 125.52 | 14.84 |
| 16:33:49 | 66.48 | 130.96 | 17.63 |
| 16:33:55 | 65.20 | 116.58 | 12.88 |
| 16:34:00 | 62.51 | 123.21 | 14.47 |
| 16:34:06 | 64.86 | 105.93 | 11.10 |
| 16:34:12 | 60.32 | 146.00 | 10.08 |
| 16:34:17 | 58.01 | 80.42 | 7.39 |
| 16:34:23 | 81.91 | 122.26 | 7.63 |
| 16:34:28 | 81.37 | 77.95 | 6.08 |
| 16:34:33 | 85.77 | 114.30 | 6.69 |
| 16:34:39 | 53.24 | 70.96 | 6.11 |
| 16:34:44 | 52.37 | 95.20 | 6.00 |
| 16:34:49 | 72.58 | 62.84 | 5.69 |
| 16:34:55 | 72.48 | 63.10 | 5.68 |
| 16:35:00 | 48.01 | 60.53 | 5.80 |
| 16:35:05 | 41.67 | 56.79 | 3.62 |

## AdaptiveK Updates (spec_decode_worker.py:689)
| Time | Transition | Reason Snapshot |
|---|---|---|
| 16:33:27 | 2 -> 3 | waste=0.1014 pos2=0.7972 pos3=0.7972 pos4=0.7972 q=143 cap=8 |
| 16:33:31 | 3 -> 2 | waste=0.2551 pos2=0.7197 pos3=0.5152 pos4=0.5152 q=132 cap=8 |
| 16:33:34 | 2 -> 3 | waste=0.1055 pos2=0.7891 pos3=0.7891 pos4=0.7891 q=128 cap=8 |
| 16:33:48 | 3 -> 4 | waste=0.1493 pos2=0.8333 pos3=0.7188 pos4=0.7188 q=96 cap=8 |
| 16:33:52 | 4 -> 3 | waste=0.2558 pos2=0.7791 pos3=0.6512 pos4=0.5465 q=86 cap=8 |
| 16:33:56 | 3 -> 4 | waste=0.1491 pos2=0.8684 pos3=0.6842 pos4=0.6842 q=76 cap=8 |
| 16:34:02 | 4 -> 3 | waste=0.2539 pos2=0.7969 pos3=0.6406 pos4=0.5469 q=64 cap=8 |
| 16:34:16 | 3 -> 2 | waste=0.2460 pos2=0.7143 pos3=0.5476 pos4=0.5476 q=42 cap=8 |

## P1P2 Timeline (spec_decode_worker.py:981)
| Time | k | step_accept | waste_ratio |
|---|---:|---|---:|
| 16:33:25 | 2 | [1.0, 0.6390] | 0.1805 |
| 16:33:38 | 3 | [1.0, 0.7566, 0.6063] | 0.2124 |
| 16:33:43 | 3 | [1.0, 0.7743, 0.6253] | 0.2001 |
| 16:33:47 | 3 | [1.0, 0.7787, 0.6334] | 0.1960 |
| 16:33:56 | 3 | [1.0, 0.8046, 0.6926] | 0.1676 |
| 16:34:01 | 4 | [1.0, 0.8260, 0.6927, 0.6064] | 0.2187 |
| 16:34:05 | 3 | [1.0, 0.8244, 0.7048] | 0.1569 |
| 16:34:09 | 3 | [1.0, 0.8111, 0.6847] | 0.1681 |
| 16:34:13 | 3 | [1.0, 0.8056, 0.6799] | 0.1715 |
| 16:34:19 | 2 | [1.0, 0.7690] | 0.1155 |
| 16:34:22 | 2 | [1.0, 0.7760] | 0.1120 |
| 16:34:25 | 2 | [1.0, 0.7746] | 0.1127 |
| 16:34:28 | 2 | [1.0, 0.7659] | 0.1170 |
| 16:34:31 | 2 | [1.0, 0.7670] | 0.1165 |
| 16:34:34 | 2 | [1.0, 0.7667] | 0.1167 |
| 16:34:37 | 2 | [1.0, 0.7611] | 0.1195 |
| 16:34:39 | 2 | [1.0, 0.7656] | 0.1172 |
| 16:34:42 | 2 | [1.0, 0.7607] | 0.1196 |
| 16:34:44 | 2 | [1.0, 0.7542] | 0.1229 |
| 16:34:47 | 2 | [1.0, 0.7533] | 0.1233 |
| 16:34:49 | 2 | [1.0, 0.7519] | 0.1240 |
| 16:34:51 | 2 | [1.0, 0.7418] | 0.1291 |
| 16:34:53 | 2 | [1.0, 0.7445] | 0.1277 |
| 16:34:55 | 2 | [1.0, 0.7469] | 0.1266 |
| 16:34:57 | 2 | [1.0, 0.7489] | 0.1255 |
| 16:34:59 | 2 | [1.0, 0.7543] | 0.1228 |
| 16:35:01 | 2 | [1.0, 0.7460] | 0.1270 |
| 16:35:03 | 2 | [1.0, 0.7480] | 0.1260 |
| 16:35:05 | 2 | [1.0, 0.7339] | 0.1330 |
| 16:35:07 | 2 | [1.0, 0.7418] | 0.1291 |

## P1P2 Trend Notes
- Adaptive process transitions `2 -> 3 -> 2 -> 3 -> 4 -> 3 -> 4 -> 3 -> 2`, then remains at `k=2`.
- At `k=3/4`, higher-order acceptance improves notably in the mid-run, but waste rises when queue depth falls.
- After converging back to `k=2`, step-2 acceptance stabilizes around `0.74-0.77` with waste mostly `0.112-0.13`.

## Consolidated Trend Notes
- Compared with the previous adaptive spec2 record, this run ends with lower throughput (`1.69 req/s`, `376.97 tok/s`) and better TTFT (`270.51 ms` mean).
- The adaptive controller explores `k=3` and `k=4` at medium queue depths, then settles to `k=2` for tail stability.
- Final speculative efficiency plateaus around draft acceptance `~0.762` and system efficiency `~0.681`.

## Source
- Source data: user-provided adaptive spec2 logs (with AdaptiveK and P1P2) and final benchmark summary in this conversation.