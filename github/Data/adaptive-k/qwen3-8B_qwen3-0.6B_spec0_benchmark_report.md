# qwen3-8B + qwen3-0.6B (speculative_tokens=0) Benchmark Report

## Run Profile
- target_model: qwen3-8B
- draft_model: qwen3-0.6B (not used in spec0 baseline)
- num_speculative_tokens: 0
- traffic_request_rate: 16.0
- burstiness_factor: 1.0 (Poisson process)
- benchmark_completion: 200/200 requests
- log_window: 13:16:08 -> 13:17:01 (provided runtime samples)

## Final Serving Benchmark Result
| Metric | Value |
|---|---:|
| Successful requests | 200 |
| Benchmark duration (s) | 63.21 |
| Total input tokens | 43560 |
| Total generated tokens | 44524 |
| Request throughput (req/s) | 3.16 |
| Output token throughput (tok/s) | 704.33 |
| Total token throughput (tok/s) | 1393.41 |
| Mean TTFT (ms) | 237.24 |
| Median TTFT (ms) | 213.33 |
| P99 TTFT (ms) | 511.24 |
| Mean TPOT (ms) | 149.94 |
| Median TPOT (ms) | 112.75 |
| P99 TPOT (ms) | 329.87 |
| Mean ITL (ms) | 94.63 |
| Median ITL (ms) | 76.32 |
| P99 ITL (ms) | 492.31 |

## Runtime Throughput Timeline (metrics.py:417)
| Time | Prompt throughput (tok/s) | Generation throughput (tok/s) | Running reqs | GPU KV cache usage |
|---|---:|---:|---:|---:|
| 13:16:08 | 0.0 | 1483.3 | 121 | 14.6% |
| 13:16:13 | 0.0 | 1386.9 | 99 | 14.5% |
| 13:16:18 | 0.0 | 1157.2 | 79 | 13.6% |
| 13:16:23 | 0.0 | 997.8 | 64 | 12.8% |
| 13:16:28 | 0.0 | 858.6 | 51 | 12.3% |
| 13:16:33 | 0.0 | 679.7 | 35 | 9.2% |
| 13:16:38 | 0.0 | 493.3 | 21 | 6.2% |
| 13:16:43 | 0.0 | 273.4 | 10 | 3.4% |
| 13:16:48 | 0.0 | 148.8 | 7 | 2.5% |
| 13:17:01 | 0.0 | 20.9 | 0 | 0.0% |

## Consolidated Trend Notes
- This spec0 baseline delivers the best aggregate throughput among provided settings so far.
- Generation throughput in runtime samples starts very high and then declines as queue depth drains.
- Queue depth and KV cache usage reduce monotonically toward zero, consistent with normal tail behavior.
- Latency metrics (TTFT/TPOT/ITL) are significantly lower than higher-spec-token settings.

## Source
- Source data: user-provided spec0 runtime logs and serving benchmark summary in this conversation.