# qwen3-8B + qwen3-0.6B (speculative_tokens=8) Benchmark Report

## Run Profile
- target_model: qwen3-8B
- draft_model: qwen3-0.6B
- num_speculative_tokens: 8
- log_window: 12:55:39 -> 12:58:54
- benchmark_completion: 200/200 requests

## Final Serving Benchmark Result
| Metric | Value |
|---|---:|
| Successful requests | 200 |
| Benchmark duration (s) | 192.27 |
| Total input tokens | 43560 |
| Total generated tokens | 44636 |
| Request throughput (req/s) | 1.04 |
| Output token throughput (tok/s) | 232.16 |
| Total token throughput (tok/s) | 458.71 |
| Mean TTFT (ms) | 567.15 |
| Median TTFT (ms) | 358.93 |
| P99 TTFT (ms) | 1967.95 |
| Mean TPOT (ms) | 479.52 |
| Median TPOT (ms) | 350.06 |
| P99 TPOT (ms) | 2021.05 |
| Mean ITL (ms) | 979.48 |
| Median ITL (ms) | 891.60 |
| P99 ITL (ms) | 5562.85 |

## Key Runtime Timeline (metrics.py:417)
| Time | Prompt throughput (tok/s) | Generation throughput (tok/s) | Running reqs | GPU KV cache usage |
|---|---:|---:|---:|---:|
| 12:55:39 | 1864.1 | 43.6 | 186 | 36.7% |
| 12:55:50 | 0.0 | 146.6 | 143 | 27.5% |
| 12:56:12 | 0.0 | 127.6 | 113 | 27.7% |
| 12:56:44 | 0.0 | 93.0 | 56 | 20.2% |
| 12:57:10 | 0.0 | 49.9 | 24 | 10.7% |
| 12:57:42 | 0.0 | 13.7 | 6 | 3.0% |
| 12:58:30 | 0.0 | 4.4 | 1 | 0.8% |
| 12:58:44 | 0.0 | 0.8 | 0 | 0.0% |
| 12:58:54 | 0.0 | 0.0 | 0 | 0.0% |

## Key Speculative Timeline (metrics.py:439)
| Time | Draft acceptance rate | System efficiency | Accepted tokens | Draft tokens | Emitted tokens |
|---|---:|---:|---:|---:|---:|
| 12:55:39 | 0.650 | 0.325 | 2115 | 3256 | 1191 |
| 12:55:50 | 0.662 | 0.348 | 6871 | 10376 | 4058 |
| 12:55:55 | 0.672 | 0.370 | 11786 | 17544 | 7297 |
| 12:56:12 | 0.693 | 0.408 | 26454 | 38176 | 17507 |
| 12:56:44 | 0.717 | 0.447 | 47100 | 65736 | 33077 |
| 12:57:10 | 0.722 | 0.456 | 57047 | 79016 | 40551 |
| 12:57:42 | 0.719 | 0.449 | 62782 | 87368 | 44143 |
| 12:58:20 | 0.717 | 0.447 | 64071 | 89328 | 44900 |
| 12:58:30 | 0.717 | 0.446 | 64417 | 89840 | 45087 |
| 12:58:44 | 0.717 | 0.446 | 64512 | 89976 | 45135 |

## Key Stage-Time Timeline (spec_decode_worker.py:1115)
| Time | avg_time_per_proposal_tok_ms | scoring_time_ms | verification_time_ms |
|---|---:|---:|---:|
| 12:55:39 | 203.10 | 1502.54 | 878.17 |
| 12:55:46 | 64.24 | 362.64 | 47.96 |
| 12:56:12 | 68.08 | 258.27 | 37.72 |
| 12:56:31 | 54.77 | 189.52 | 28.44 |
| 12:56:49 | 49.94 | 123.81 | 18.40 |
| 12:57:10 | 46.96 | 98.95 | 13.54 |
| 12:57:30 | 42.68 | 76.97 | 7.25 |
| 12:57:48 | 63.08 | 86.41 | 5.71 |
| 12:58:11 | 70.83 | 95.74 | 5.74 |
| 12:58:34 | 40.83 | 58.00 | 3.64 |

## P1P2 Timeline (spec_decode_worker.py:857, k=8)
| Time | k | step_accept | waste_ratio |
|---|---:|---|---:|
| 14:06:26 | 8 | [1.0, 0.6085, 0.4780, 0.2983, 0.1904, 0.1215, 0.0566, 0.0530] | 0.6492 |
| 14:06:37 | 8 | [1.0, 0.6219, 0.4790, 0.3088, 0.2047, 0.1387, 0.0782, 0.0695] | 0.6374 |
| 14:06:47 | 8 | [1.0, 0.6308, 0.4840, 0.3207, 0.2210, 0.1565, 0.0996, 0.0884] | 0.6249 |
| 14:06:57 | 8 | [1.0, 0.6420, 0.4928, 0.3357, 0.2403, 0.1772, 0.1235, 0.1097] | 0.6098 |
| 14:07:06 | 8 | [1.0, 0.6532, 0.5027, 0.3508, 0.2579, 0.1960, 0.1455, 0.1304] | 0.5954 |
| 14:07:14 | 8 | [1.0, 0.6639, 0.5121, 0.3634, 0.2743, 0.2132, 0.1643, 0.1477] | 0.5826 |
| 14:07:22 | 8 | [1.0, 0.6724, 0.5198, 0.3760, 0.2891, 0.2285, 0.1792, 0.1626] | 0.5716 |
| 14:07:29 | 8 | [1.0, 0.6803, 0.5271, 0.3878, 0.3030, 0.2431, 0.1939, 0.1763] | 0.5611 |
| 14:07:36 | 8 | [1.0, 0.6883, 0.5344, 0.3987, 0.3139, 0.2552, 0.2072, 0.1894] | 0.5516 |
| 14:07:42 | 8 | [1.0, 0.6934, 0.5395, 0.4061, 0.3222, 0.2626, 0.2173, 0.1989] | 0.5450 |
| 14:07:48 | 8 | [1.0, 0.6965, 0.5429, 0.4106, 0.3263, 0.2672, 0.2226, 0.2036] | 0.5413 |
| 14:07:53 | 8 | [1.0, 0.6994, 0.5436, 0.4133, 0.3286, 0.2701, 0.2244, 0.2040] | 0.5396 |
| 14:07:59 | 8 | [1.0, 0.7013, 0.5455, 0.4122, 0.3297, 0.2714, 0.2258, 0.2045] | 0.5387 |
| 14:08:05 | 8 | [1.0, 0.7015, 0.5397, 0.4078, 0.3245, 0.2668, 0.2232, 0.2003] | 0.5420 |
| 14:08:12 | 8 | [1.0, 0.6990, 0.5346, 0.4020, 0.3190, 0.2595, 0.2164, 0.1935] | 0.5470 |
| 14:08:19 | 8 | [1.0, 0.6992, 0.5324, 0.3989, 0.3175, 0.2578, 0.2149, 0.1912] | 0.5485 |
| 14:08:25 | 8 | [1.0, 0.6989, 0.5321, 0.3992, 0.3185, 0.2593, 0.2160, 0.1919] | 0.5480 |
| 14:08:31 | 8 | [1.0, 0.6988, 0.5285, 0.3948, 0.3156, 0.2572, 0.2129, 0.1884] | 0.5505 |
| 14:08:37 | 8 | [1.0, 0.7011, 0.5337, 0.3970, 0.3163, 0.2578, 0.2139, 0.1879] | 0.5490 |
| 14:08:44 | 8 | [1.0, 0.7002, 0.5306, 0.3880, 0.3098, 0.2521, 0.2075, 0.1825] | 0.5537 |
| 14:08:51 | 8 | [1.0, 0.7028, 0.5331, 0.3863, 0.3092, 0.2517, 0.2056, 0.1808] | 0.5538 |
| 14:08:57 | 8 | [1.0, 0.7002, 0.5344, 0.3868, 0.3101, 0.2510, 0.2015, 0.1738] | 0.5553 |
| 14:09:04 | 8 | [1.0, 0.6989, 0.5306, 0.3848, 0.3060, 0.2466, 0.1977, 0.1686] | 0.5583 |
| 14:09:10 | 8 | [1.0, 0.7001, 0.5355, 0.3830, 0.3035, 0.2449, 0.1978, 0.1686] | 0.5583 |
| 14:09:17 | 8 | [1.0, 0.7013, 0.5342, 0.3801, 0.3010, 0.2411, 0.1933, 0.1639] | 0.5606 |
| 14:09:23 | 8 | [1.0, 0.6979, 0.5270, 0.3769, 0.2993, 0.2403, 0.1918, 0.1612] | 0.5632 |
| 14:09:27 | 8 | [1.0, 0.6980, 0.5261, 0.3712, 0.2929, 0.2358, 0.1889, 0.1592] | 0.5660 |

## P1P2 Trend Notes
- `k=8` throughout the run.
- Warm-up phase shows strong improvement on deeper speculative positions.
  For example, position-8 acceptance rises from `0.0530` to about `0.20+` in mid-run.
- `waste_ratio` drops from `0.6492` to a minimum `0.5387`, then gradually rebounds to `0.5660` in the tail.
- Near the tail, later-position acceptance declines while queue depth is very small, consistent with unstable tail behavior.
## Consolidated Trend Notes
- System warms up with very high stage costs at 12:55:39, then quickly drops to a lower steady-state range.
- Draft acceptance rate increases from 0.650 to ~0.717 and plateaus near the tail.
- System efficiency increases from 0.325 to ~0.446 and then stabilizes.
- Generation throughput peaks early (146.6 tok/s) and then decays as running requests drain to zero.
- Tail phase shows low generation throughput with near-zero queue depth and very low KV cache usage.

## Source
- Source data: user-provided spec8 runtime logs and benchmark summary in this conversation.