#!/usr/bin/env python3
"""Parse vLLM spec-decode logs and surface bottleneck signals.

Usage:
  python scripts/spec_decode_diagnose.py --log /path/to/server.log
  python scripts/spec_decode_diagnose.py --log /path/to/server.log --out-dir /tmp/diag
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Iterable, List, Optional


TS_RE = re.compile(r"INFO\s+\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})")
RUNTIME_RE = re.compile(
    r"Avg prompt throughput:\s*([0-9.]+)\s+tokens/s,\s*"
    r"Avg generation throughput:\s*([0-9.]+)\s+tokens/s,\s*"
    r"Running:\s*([0-9]+)\s+reqs,.*?"
    r"GPU KV cache usage:\s*([0-9.]+)%")
SPEC_RE = re.compile(
    r"Draft acceptance rate:\s*([0-9.]+),\s*"
    r"System efficiency:\s*([0-9.]+),\s*"
    r"Number of speculative tokens:\s*([0-9]+),\s*"
    r"Number of accepted tokens:\s*([0-9]+),\s*"
    r"Number of draft tokens:\s*([0-9]+),\s*"
    r"Number of emitted tokens:\s*([0-9]+)")
STAGE_RE = re.compile(
    r"SpecDecodeWorker stage times:\s*"
    r"average_time_per_proposal_tok_ms=([0-9.]+)\s+"
    r"scoring_time_ms=([0-9.]+)\s+"
    r"verification_time_ms=([0-9.]+)")
P12_RE = re.compile(
    r"P1P2 k=([0-9]+)\s+step_accept=\[([^\]]*)\]\s+waste_ratio=([0-9.]+)")
ALIGN_RE = re.compile(
    r"ALIGN P1/P2 hs_ratio=([0-9.]+)\s+accept=([0-9.]+)\s+"
    r"accept_with_hs=([0-9.]+)\s+accept_no_hs=([0-9.]+)\s+"
    r"throughput_tps=([0-9.]+)\s+iter_p95=([0-9.]+)")
ADP_STATS_RE = re.compile(
    r"AdaptiveK stats switch_per_min=([0-9.]+)\s+high_k_occ=([0-9.]+)\s+hist=(.+)")
FASTPATH_RE = re.compile(
    r"DraftGPUFastPath hit_rate=([0-9.]+)\s+total=([0-9]+)\s+hit=([0-9]+)\s+"
    r"fail_disabled=([0-9]+)\s+fail_prompt=([0-9]+)\s+"
    r"fail_backend=([0-9]+)\s+fail_lora=([0-9]+)\s+fail_adapter=([0-9]+)")


@dataclass
class RuntimeSample:
    timestamp: str
    prompt_tps: float
    gen_tps: float
    running_reqs: int
    gpu_kv_pct: float


@dataclass
class SpecSample:
    timestamp: str
    k: int
    acceptance_rate: float
    system_efficiency: float
    accepted_tokens: int
    draft_tokens: int
    emitted_tokens: int


@dataclass
class StageSample:
    timestamp: str
    k_hint: Optional[int]
    proposal_ms: float
    scoring_ms: float
    verification_ms: float

    @property
    def total_ms(self) -> float:
        return self.proposal_ms + self.scoring_ms + self.verification_ms


@dataclass
class P12Sample:
    timestamp: str
    k: int
    step_accept: List[float]
    waste_ratio: float


@dataclass
class AlignSample:
    timestamp: str
    hs_ratio: float
    accept: float
    accept_with_hs: float
    accept_no_hs: float
    throughput_tps: float
    iter_p95_ms: float


def _percentile(values: Iterable[float], pct: float) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        return 0.0
    idx = min(len(arr) - 1, int((pct / 100.0) * (len(arr) - 1)))
    return arr[idx]


def _safe_mean(values: Iterable[float]) -> float:
    arr = [float(v) for v in values]
    return mean(arr) if arr else 0.0


def _parse_step_accept(raw: str) -> List[float]:
    raw = raw.strip()
    if not raw:
        return []
    out: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def _segment(values: List[StageSample]) -> tuple[List[StageSample], List[StageSample], List[StageSample]]:
    n = len(values)
    if n == 0:
        return [], [], []
    if n < 6:
        one = max(1, n // 3)
        return values[:one], values[one:-one] if n - 2 * one > 0 else [], values[-one:]
    one = n // 3
    return values[:one], values[one:2 * one], values[2 * one:]


def parse_log(log_path: str):
    runtime_samples: List[RuntimeSample] = []
    spec_samples: List[SpecSample] = []
    stage_samples: List[StageSample] = []
    p12_samples: List[P12Sample] = []
    align_samples: List[AlignSample] = []
    adaptive_stats = None
    fastpath_stats = None

    last_spec_k: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts_m = TS_RE.search(line)
            ts = ts_m.group(1) if ts_m else ""

            m = RUNTIME_RE.search(line)
            if m:
                runtime_samples.append(
                    RuntimeSample(
                        timestamp=ts,
                        prompt_tps=float(m.group(1)),
                        gen_tps=float(m.group(2)),
                        running_reqs=int(m.group(3)),
                        gpu_kv_pct=float(m.group(4)),
                    ))
                continue

            m = SPEC_RE.search(line)
            if m:
                k = int(m.group(3))
                last_spec_k = k
                spec_samples.append(
                    SpecSample(
                        timestamp=ts,
                        acceptance_rate=float(m.group(1)),
                        system_efficiency=float(m.group(2)),
                        k=k,
                        accepted_tokens=int(m.group(4)),
                        draft_tokens=int(m.group(5)),
                        emitted_tokens=int(m.group(6)),
                    ))
                continue

            m = STAGE_RE.search(line)
            if m:
                stage_samples.append(
                    StageSample(
                        timestamp=ts,
                        k_hint=last_spec_k,
                        proposal_ms=float(m.group(1)),
                        scoring_ms=float(m.group(2)),
                        verification_ms=float(m.group(3)),
                    ))
                continue

            m = P12_RE.search(line)
            if m:
                p12_samples.append(
                    P12Sample(
                        timestamp=ts,
                        k=int(m.group(1)),
                        step_accept=_parse_step_accept(m.group(2)),
                        waste_ratio=float(m.group(3)),
                    ))
                continue

            m = ALIGN_RE.search(line)
            if m:
                align_samples.append(
                    AlignSample(
                        timestamp=ts,
                        hs_ratio=float(m.group(1)),
                        accept=float(m.group(2)),
                        accept_with_hs=float(m.group(3)),
                        accept_no_hs=float(m.group(4)),
                        throughput_tps=float(m.group(5)),
                        iter_p95_ms=float(m.group(6)),
                    ))
                continue

            m = ADP_STATS_RE.search(line)
            if m:
                adaptive_stats = {
                    "switch_per_min": float(m.group(1)),
                    "high_k_occ": float(m.group(2)),
                    "hist": m.group(3).strip(),
                }
                continue

            m = FASTPATH_RE.search(line)
            if m:
                fastpath_stats = {
                    "hit_rate": float(m.group(1)),
                    "total": int(m.group(2)),
                    "hit": int(m.group(3)),
                    "fail_disabled": int(m.group(4)),
                    "fail_prompt": int(m.group(5)),
                    "fail_backend": int(m.group(6)),
                    "fail_lora": int(m.group(7)),
                    "fail_adapter": int(m.group(8)),
                }

    k_hist = Counter(s.k for s in spec_samples)
    k_hist_text = ",".join(f"{k}:{k_hist[k]}" for k in sorted(k_hist)) if k_hist else ""

    warmup, steady, tail = _segment(stage_samples)

    summary = {
        "log_path": os.path.abspath(log_path),
        "runtime_points": len(runtime_samples),
        "spec_points": len(spec_samples),
        "stage_points": len(stage_samples),
        "p12_points": len(p12_samples),
        "align_points": len(align_samples),
        "gen_tps_mean": _safe_mean(s.gen_tps for s in runtime_samples),
        "gen_tps_peak": max((s.gen_tps for s in runtime_samples), default=0.0),
        "running_reqs_peak": max((s.running_reqs for s in runtime_samples), default=0),
        "gpu_kv_peak_pct": max((s.gpu_kv_pct for s in runtime_samples), default=0.0),
        "accept_last": spec_samples[-1].acceptance_rate if spec_samples else 0.0,
        "eff_last": spec_samples[-1].system_efficiency if spec_samples else 0.0,
        "k_last": spec_samples[-1].k if spec_samples else 0,
        "k_hist": dict(sorted(k_hist.items())),
        "k_hist_text": k_hist_text,
        "stage_proposal_mean_ms": _safe_mean(s.proposal_ms for s in stage_samples),
        "stage_proposal_p95_ms": _percentile((s.proposal_ms for s in stage_samples), 95),
        "stage_scoring_mean_ms": _safe_mean(s.scoring_ms for s in stage_samples),
        "stage_scoring_p95_ms": _percentile((s.scoring_ms for s in stage_samples), 95),
        "stage_verify_mean_ms": _safe_mean(s.verification_ms for s in stage_samples),
        "stage_verify_p95_ms": _percentile((s.verification_ms for s in stage_samples), 95),
        "stage_total_mean_ms": _safe_mean(s.total_ms for s in stage_samples),
        "stage_total_p95_ms": _percentile((s.total_ms for s in stage_samples), 95),
        "stage_scoring_share_pct": 0.0,
        "stage_verify_share_pct": 0.0,
        "warmup_scoring_mean_ms": _safe_mean(s.scoring_ms for s in warmup),
        "steady_scoring_mean_ms": _safe_mean(s.scoring_ms for s in steady),
        "tail_scoring_mean_ms": _safe_mean(s.scoring_ms for s in tail),
        "warmup_verify_mean_ms": _safe_mean(s.verification_ms for s in warmup),
        "steady_verify_mean_ms": _safe_mean(s.verification_ms for s in steady),
        "tail_verify_mean_ms": _safe_mean(s.verification_ms for s in tail),
        "p12_waste_mean": _safe_mean(s.waste_ratio for s in p12_samples),
        "p12_waste_last": p12_samples[-1].waste_ratio if p12_samples else 0.0,
        "p12_last_k": p12_samples[-1].k if p12_samples else 0,
        "p12_last_step_accept": p12_samples[-1].step_accept if p12_samples else [],
        "align_accept_last": align_samples[-1].accept if align_samples else 0.0,
        "align_iter_p95_last": align_samples[-1].iter_p95_ms if align_samples else 0.0,
        "adaptive_stats": adaptive_stats,
        "fastpath_stats": fastpath_stats,
    }

    total_stage = (
        summary["stage_proposal_mean_ms"] + summary["stage_scoring_mean_ms"] +
        summary["stage_verify_mean_ms"])
    if total_stage > 0:
        summary["stage_scoring_share_pct"] = summary["stage_scoring_mean_ms"] * 100.0 / total_stage
        summary["stage_verify_share_pct"] = summary["stage_verify_mean_ms"] * 100.0 / total_stage

    return summary, runtime_samples, spec_samples, stage_samples, p12_samples, align_samples


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict) -> str:
    return f"""# Spec Decode Diagnosis

## Key Signals
| Metric | Value |
|---|---:|
| gen_tps_mean | {summary["gen_tps_mean"]:.2f} |
| gen_tps_peak | {summary["gen_tps_peak"]:.2f} |
| accept_last | {summary["accept_last"]:.3f} |
| eff_last | {summary["eff_last"]:.3f} |
| k_last | {summary["k_last"]} |
| k_hist | {summary["k_hist_text"]} |

## Stage Breakdown
| Metric | Value |
|---|---:|
| proposal_mean_ms | {summary["stage_proposal_mean_ms"]:.2f} |
| scoring_mean_ms | {summary["stage_scoring_mean_ms"]:.2f} |
| verify_mean_ms | {summary["stage_verify_mean_ms"]:.2f} |
| total_mean_ms | {summary["stage_total_mean_ms"]:.2f} |
| scoring_share_pct | {summary["stage_scoring_share_pct"]:.1f} |
| verify_share_pct | {summary["stage_verify_share_pct"]:.1f} |
| scoring_p95_ms | {summary["stage_scoring_p95_ms"]:.2f} |
| verify_p95_ms | {summary["stage_verify_p95_ms"]:.2f} |

## Warmup vs Steady
| Metric | Warmup | Steady | Tail |
|---|---:|---:|---:|
| scoring_mean_ms | {summary["warmup_scoring_mean_ms"]:.2f} | {summary["steady_scoring_mean_ms"]:.2f} | {summary["tail_scoring_mean_ms"]:.2f} |
| verify_mean_ms | {summary["warmup_verify_mean_ms"]:.2f} | {summary["steady_verify_mean_ms"]:.2f} | {summary["tail_verify_mean_ms"]:.2f} |

## P1P2
| Metric | Value |
|---|---:|
| p12_waste_mean | {summary["p12_waste_mean"]:.4f} |
| p12_waste_last | {summary["p12_waste_last"]:.4f} |
| p12_last_k | {summary["p12_last_k"]} |
| p12_last_step_accept | {summary["p12_last_step_accept"]} |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to server log")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory for JSON/CSV/MD outputs. If omitted, print summary only.",
    )
    args = parser.parse_args()

    summary, runtime_samples, spec_samples, stage_samples, p12_samples, align_samples = parse_log(
        args.log)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.out_dir:
        return

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    write_csv(
        os.path.join(out_dir, "runtime_timeseries.csv"),
        [asdict(s) for s in runtime_samples],
    )
    write_csv(
        os.path.join(out_dir, "spec_timeseries.csv"),
        [asdict(s) for s in spec_samples],
    )
    stage_rows = []
    for s in stage_samples:
        row = asdict(s)
        row["total_ms"] = s.total_ms
        stage_rows.append(row)
    write_csv(os.path.join(out_dir, "stage_timeseries.csv"), stage_rows)
    write_csv(os.path.join(out_dir, "p12_timeseries.csv"), [asdict(s) for s in p12_samples])
    write_csv(os.path.join(out_dir, "align_timeseries.csv"), [asdict(s) for s in align_samples])

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(summary))


if __name__ == "__main__":
    main()
