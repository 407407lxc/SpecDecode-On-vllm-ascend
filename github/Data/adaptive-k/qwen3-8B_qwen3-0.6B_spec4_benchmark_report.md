# qwen3-8B + qwen3-0.6B (speculative_tokens=4) Benchmark Report

## Run Profile
- target_model: qwen3-8B
- draft_model: qwen3-0.6B
- num_speculative_tokens: 4
- traffic_request_rate: 16.0
- burstiness_factor: 1.0 (Poisson process)
- benchmark_completion: 200/200 requests
- log_window: 13:03:10 -> 13:05:35

## Final Serving Benchmark Result
| Metric | Value |
|---|---:|
| Successful requests | 200 |
| Benchmark duration (s) | 138.66 |
| Total input tokens | 43560 |
| Total generated tokens | 44697 |
| Request throughput (req/s) | 1.44 |
| Output token throughput (tok/s) | 322.34 |
| Total token throughput (tok/s) | 636.48 |
| Mean TTFT (ms) | 321.05 |
| Median TTFT (ms) | 255.22 |
| P99 TTFT (ms) | 1166.47 |
| Mean TPOT (ms) | 298.63 |
| Median TPOT (ms) | 233.32 |
| P99 TPOT (ms) | 912.29 |
| Mean ITL (ms) | 575.14 |
| Median ITL (ms) | 496.17 |
| P99 ITL (ms) | 2777.87 |

## Key Runtime Timeline (metrics.py:417)
| Time | Prompt throughput (tok/s) | Generation throughput (tok/s) | Running reqs | GPU KV cache usage |
|---|---:|---:|---:|---:|
| 13:03:12 | 1947.2 | 135.1 | 172 | 33.0% |
| 13:03:17 | 0.0 | 235.2 | 145 | 27.9% |
| 13:03:22 | 0.0 | 242.1 | 133 | 26.4% |
| 13:03:33 | 0.0 | 228.7 | 113 | 27.7% |
| 13:03:59 | 0.0 | 153.8 | 56 | 22.6% |
| 13:04:14 | 0.0 | 84.7 | 34 | 16.7% |
| 13:04:35 | 0.0 | 40.6 | 14 | 7.3% |
| 13:04:45 | 0.0 | 17.0 | 4 | 2.3% |
| 13:05:11 | 0.0 | 6.2 | 2 | 1.1% |
| 13:05:25 | 0.0 | 1.3 | 0 | 0.0% |
| 13:05:35 | 0.0 | 0.0 | 0 | 0.0% |

## Key Speculative Timeline (metrics.py:439)
| Time | Draft acceptance rate | System efficiency | Accepted tokens | Draft tokens | Emitted tokens |
|---|---:|---:|---:|---:|---:|
| 13:03:12 | 0.656 | 0.526 | 1403 | 2140 | 1408 |
| 13:03:22 | 0.684 | 0.567 | 9405 | 13760 | 9759 |
| 13:03:38 | 0.709 | 0.600 | 17126 | 24148 | 18120 |
| 13:03:59 | 0.740 | 0.640 | 30122 | 40700 | 32578 |
| 13:04:14 | 0.749 | 0.651 | 36445 | 48684 | 39631 |
| 13:04:29 | 0.751 | 0.654 | 38841 | 51752 | 42295 |
| 13:04:45 | 0.748 | 0.650 | 40609 | 54272 | 44089 |
| 13:05:01 | 0.747 | 0.649 | 41102 | 54992 | 44592 |
| 13:05:11 | 0.747 | 0.648 | 41331 | 55316 | 44828 |
| 13:05:25 | 0.747 | 0.648 | 41410 | 55440 | 44915 |

## Key Stage-Time Timeline (spec_decode_worker.py:1115)
| Time | avg_time_per_proposal_tok_ms | scoring_time_ms | verification_time_ms |
|---|---:|---:|---:|
| 13:03:10 | 79.51 | 255.25 | 32.00 |
| 13:03:22 | 71.65 | 171.92 | 26.92 |
| 13:03:40 | 65.29 | 143.06 | 18.04 |
| 13:04:02 | 52.12 | 110.23 | 11.51 |
| 13:04:19 | 70.41 | 115.91 | 9.21 |
| 13:04:30 | 45.77 | 97.50 | 6.68 |
| 13:04:47 | 47.23 | 62.00 | 6.13 |
| 13:05:04 | 71.41 | 89.90 | 6.38 |
| 13:05:14 | 40.35 | 56.40 | 3.55 |

## P1P2 Timeline (spec_decode_worker.py:857, k=4)
| Time | k | step_accept | waste_ratio |
|---|---:|---|---:|
| 14:00:21 | 4 | [1.0, 0.5808, 0.4714, 0.2917] | 0.4140 |
| 14:00:27 | 4 | [1.0, 0.5996, 0.4764, 0.3053] | 0.4047 |
| 14:00:33 | 4 | [1.0, 0.6144, 0.4854, 0.3226] | 0.3944 |
| 14:00:38 | 4 | [1.0, 0.6293, 0.4967, 0.3390] | 0.3837 |
| 14:00:43 | 4 | [1.0, 0.6442, 0.5120, 0.3590] | 0.3712 |
| 14:00:48 | 4 | [1.0, 0.6570, 0.5242, 0.3757] | 0.3608 |
| 14:00:53 | 4 | [1.0, 0.6691, 0.5369, 0.3909] | 0.3508 |
| 14:00:57 | 4 | [1.0, 0.6795, 0.5471, 0.4055] | 0.3420 |
| 14:01:02 | 4 | [1.0, 0.6889, 0.5556, 0.4163] | 0.3348 |
| 14:01:06 | 4 | [1.0, 0.6969, 0.5650, 0.4295] | 0.3272 |
| 14:01:09 | 4 | [1.0, 0.7031, 0.5706, 0.4381] | 0.3220 |
| 14:01:13 | 4 | [1.0, 0.7094, 0.5768, 0.4468] | 0.3167 |
| 14:01:17 | 4 | [1.0, 0.7146, 0.5786, 0.4501] | 0.3142 |
| 14:01:22 | 4 | [1.0, 0.7180, 0.5807, 0.4515] | 0.3124 |
| 14:01:27 | 4 | [1.0, 0.7184, 0.5806, 0.4523] | 0.3122 |
| 14:01:31 | 4 | [1.0, 0.7192, 0.5779, 0.4505] | 0.3131 |
| 14:01:35 | 4 | [1.0, 0.7213, 0.5772, 0.4481] | 0.3134 |
| 14:01:39 | 4 | [1.0, 0.7203, 0.5750, 0.4474] | 0.3143 |
| 14:01:43 | 4 | [1.0, 0.7183, 0.5733, 0.4464] | 0.3155 |
| 14:01:47 | 4 | [1.0, 0.7173, 0.5680, 0.4426] | 0.3180 |
| 14:01:50 | 4 | [1.0, 0.7176, 0.5655, 0.4426] | 0.3186 |
| 14:01:54 | 4 | [1.0, 0.7166, 0.5659, 0.4386] | 0.3197 |
| 14:01:58 | 4 | [1.0, 0.7132, 0.5578, 0.4315] | 0.3244 |
| 14:02:02 | 4 | [1.0, 0.7072, 0.5524, 0.4256] | 0.3287 |
| 14:02:05 | 4 | [1.0, 0.7103, 0.5553, 0.4261] | 0.3271 |
| 14:02:09 | 4 | [1.0, 0.7112, 0.5561, 0.4265] | 0.3265 |
| 14:02:13 | 4 | [1.0, 0.7115, 0.5556, 0.4251] | 0.3270 |
| 14:02:16 | 4 | [1.0, 0.7092, 0.5494, 0.4213] | 0.3300 |
| 14:02:20 | 4 | [1.0, 0.7025, 0.5413, 0.4155] | 0.3352 |
| 14:02:24 | 4 | [1.0, 0.6977, 0.5353, 0.4094] | 0.3394 |
| 14:02:26 | 4 | [1.0, 0.6992, 0.5339, 0.4105] | 0.3391 |
| 14:02:29 | 4 | [1.0, 0.6857, 0.5222, 0.4021] | 0.3475 |
| 14:02:31 | 4 | [1.0, 0.6808, 0.5137, 0.3941] | 0.3528 |
| 14:02:33 | 4 | [1.0, 0.6736, 0.5057, 0.3840] | 0.3592 |

## P1P2 Trend Notes
- `k=4` throughout this run.
- Acceptance on later speculative positions improved substantially during warm-up.
  Position-2: `0.5808 -> 0.7213`, Position-3: `0.4714 -> 0.5807`, Position-4: `0.2917 -> 0.4523`.
- `waste_ratio` dropped from `0.4140` to a minimum of `0.3122`, then rebounded in the tail to `0.3592` as queue depth became very small.
## Consolidated Trend Notes
- Draft acceptance rate rose from 0.656 to ~0.747 and then stabilized.
- System efficiency improved from 0.526 to ~0.648 and remained stable in the tail.
- Generation throughput peaked early (242.1 tok/s at 13:03:22), then declined as active requests drained.
- Running requests and GPU KV cache usage decreased steadily to zero by run end.
- Stage verification latency dropped from 32.00 ms to low single-digit milliseconds.

## Source
- Source data: user-provided spec4 runtime logs and serving benchmark summary in this conversation.