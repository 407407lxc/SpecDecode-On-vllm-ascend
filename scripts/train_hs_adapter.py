#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train HS adapter from extracted shard data.

Input shards are produced by scripts/hs_extract_alignment_data.py and contain
at least:
  - previous_hidden_states: teacher hidden states
  - student_hidden_states: student hidden states
  - sample_row_id (optional): sample id for split by request

This script trains a linear projection:
    student_hs ~= Linear(teacher_hs)

and exports adapter weights in a format directly loadable by
vllm-ascend/vllm_ascend/models/qwen3.py:
    {"weight": tensor[out_dim, in_dim], "bias": tensor[out_dim]}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


LOGGER = logging.getLogger("train_hs_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HS adapter from extracted shard files.")
    parser.add_argument("--data-dir",
                        type=Path,
                        required=True,
                        help="Directory containing hs_align_shard_*.pt")
    parser.add_argument("--shard-pattern",
                        type=str,
                        default="hs_align_shard_*.pt",
                        help="Glob pattern for shard files")
    parser.add_argument("--output-path",
                        type=Path,
                        required=True,
                        help="Path to save adapter .pt")
    parser.add_argument("--metrics-path",
                        type=Path,
                        default=None,
                        help="Optional path to save metrics JSON")

    parser.add_argument("--input-key",
                        type=str,
                        default="previous_hidden_states",
                        help="Input tensor key in shards")
    parser.add_argument("--target-key",
                        type=str,
                        default="student_hidden_states",
                        help="Target tensor key in shards")
    parser.add_argument("--sample-id-key",
                        type=str,
                        default="sample_row_id",
                        help="Sample id key for request-level split")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bias",
                        action="store_true",
                        default=True,
                        help="Enable linear bias")
    parser.add_argument("--no-bias",
                        action="store_false",
                        dest="bias",
                        help="Disable linear bias")

    parser.add_argument("--mse-weight",
                        type=float,
                        default=1.0,
                        help="Weight of MSE loss")
    parser.add_argument("--cos-weight",
                        type=float,
                        default=0.1,
                        help="Weight of cosine loss (1-cos)")

    parser.add_argument("--val-ratio",
                        type=float,
                        default=0.05,
                        help="Validation ratio in [0,1)")
    parser.add_argument("--split-by-sample-id",
                        action="store_true",
                        default=True,
                        help="Split by sample id hash to avoid leakage")
    parser.add_argument("--no-split-by-sample-id",
                        action="store_false",
                        dest="split_by_sample_id")

    parser.add_argument("--input-rms-norm",
                        action="store_true",
                        default=True,
                        help="Apply RMS normalization to input hidden states")
    parser.add_argument("--no-input-rms-norm",
                        action="store_false",
                        dest="input_rms_norm")
    parser.add_argument("--norm-eps", type=float, default=1e-6)

    parser.add_argument("--max-rows",
                        type=int,
                        default=0,
                        help="Max rows to consume across all shards; 0 means all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        help="auto|cpu|cuda[:id]|npu[:id]")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--log-level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        parts = device_str.split(":")
        if len(parts) == 1:
            return torch.device(parts[0])
        if len(parts) == 2:
            return torch.device(parts[0], int(parts[1]))
        raise ValueError(f"Invalid --device: {device_str}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None:
        try:
            if torch.npu.is_available():
                return torch.device("npu")
        except Exception:  # pylint: disable=broad-except
            pass
    return torch.device("cpu")


def maybe_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = x.pow(2).mean(dim=-1, keepdim=True).add_(eps).sqrt_()
    return x / denom


def list_shards(data_dir: Path, pattern: str) -> List[Path]:
    shards = sorted(data_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No shards found in {data_dir} with pattern {pattern}")
    return shards


def load_shard_tensors(path: Path, input_key: str, target_key: str,
                       sample_id_key: str) -> Tuple[torch.Tensor, torch.Tensor,
                                                     Optional[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "tensors" in payload:
        tensors = payload["tensors"]
    elif isinstance(payload, dict):
        tensors = payload
    else:
        raise ValueError(f"Unexpected shard payload type for {path}")

    if input_key not in tensors:
        raise KeyError(f"Missing input key {input_key!r} in {path}")
    if target_key not in tensors:
        raise KeyError(f"Missing target key {target_key!r} in {path}")

    x = tensors[input_key]
    y = tensors[target_key]
    sid = tensors.get(sample_id_key)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"Input/target must be 2D, got {tuple(x.shape)} and {tuple(y.shape)} in {path}"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch: x={tuple(x.shape)} y={tuple(y.shape)} in {path}")

    return x, y, sid


def split_mask(num_rows: int,
               sample_ids: Optional[torch.Tensor],
               val_ratio: float,
               seed: int,
               shard_index: int,
               split_by_sample_id: bool) -> torch.Tensor:
    if val_ratio <= 0:
        return torch.zeros(num_rows, dtype=torch.bool)

    if split_by_sample_id and sample_ids is not None:
        sid = sample_ids.to(torch.int64).flatten()
        if sid.numel() == num_rows:
            hash_u = (sid * 6364136223846793005 + 1442695040888963407 +
                      int(seed)) & 0x7FFFFFFFFFFFFFFF
            threshold = int(val_ratio * 10000)
            return (hash_u % 10000) < threshold

        LOGGER.warning(
            "sample_id rows mismatch (sid=%d rows=%d), fallback to random split",
            sid.numel(),
            num_rows,
        )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + shard_index * 9973)
    return torch.rand(num_rows, generator=gen) < val_ratio


def iterate_batches(indices: torch.Tensor, batch_size: int,
                    seed: int) -> Sequence[torch.Tensor]:
    if indices.numel() == 0:
        return []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(indices.numel(), generator=gen)
    shuffled = indices[perm]
    return [shuffled[i:i + batch_size] for i in range(0, shuffled.numel(), batch_size)]


@dataclass
class LossMeter:
    rows: int = 0
    loss_sum: float = 0.0
    mse_sum: float = 0.0
    cos_sum: float = 0.0

    def update(self, loss: float, mse: float, cos: float, rows: int) -> None:
        self.rows += rows
        self.loss_sum += loss * rows
        self.mse_sum += mse * rows
        self.cos_sum += cos * rows

    def as_dict(self) -> Dict[str, float]:
        if self.rows == 0:
            return {
                "rows": 0,
                "loss": 0.0,
                "mse": 0.0,
                "cos": 0.0,
            }
        inv = 1.0 / self.rows
        return {
            "rows": self.rows,
            "loss": self.loss_sum * inv,
            "mse": self.mse_sum * inv,
            "cos": self.cos_sum * inv,
        }


def compute_loss(pred: torch.Tensor, target: torch.Tensor, mse_w: float,
                 cos_w: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred, target)
    if cos_w > 0:
        cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    else:
        cos = pred.new_zeros(())
    loss = mse_w * mse + cos_w * cos
    return loss, mse, cos


def train_epoch(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                shards: List[Path],
                args: argparse.Namespace,
                device: torch.device,
                epoch: int,
                max_rows_state: Dict[str, int]) -> LossMeter:
    model.train()
    meter = LossMeter()
    step = 0

    shard_order = list(shards)
    random.Random(args.seed + epoch).shuffle(shard_order)

    for shard_idx, shard_path in enumerate(shard_order):
        x, y, sid = load_shard_tensors(shard_path, args.input_key,
                                       args.target_key, args.sample_id_key)

        if args.max_rows > 0:
            remain = args.max_rows - max_rows_state["seen_rows_train"]
            if remain <= 0:
                break
            take = min(remain, x.shape[0])
            x = x[:take]
            y = y[:take]
            if sid is not None:
                sid = sid[:take]
            max_rows_state["seen_rows_train"] += take

        val_mask = split_mask(x.shape[0], sid, args.val_ratio, args.seed,
                              shard_idx, args.split_by_sample_id)
        train_idx = (~val_mask).nonzero(as_tuple=False).squeeze(-1)
        if train_idx.numel() == 0:
            continue

        batches = iterate_batches(train_idx, args.batch_size,
                                  args.seed + epoch * 100003 + shard_idx)
        for batch_idx in batches:
            bx = x.index_select(0, batch_idx).float().to(device)
            by = y.index_select(0, batch_idx).float().to(device)

            if args.input_rms_norm:
                bx = maybe_rms_norm(bx, args.norm_eps)

            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss, mse, cos = compute_loss(pred, by, args.mse_weight,
                                          args.cos_weight)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            rows = bx.shape[0]
            meter.update(float(loss.detach().cpu()), float(mse.detach().cpu()),
                         float(cos.detach().cpu()), rows)

            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                m = meter.as_dict()
                LOGGER.info(
                    "train epoch=%d step=%d rows=%d loss=%.6f mse=%.6f cos=%.6f",
                    epoch,
                    step,
                    int(m["rows"]),
                    m["loss"],
                    m["mse"],
                    m["cos"],
                )

        del x, y, sid

    return meter


def eval_epoch(model: nn.Module, shards: List[Path], args: argparse.Namespace,
               device: torch.device, epoch: int,
               max_rows_state: Dict[str, int]) -> LossMeter:
    if args.val_ratio <= 0:
        return LossMeter()

    model.eval()
    meter = LossMeter()

    with torch.inference_mode():
        for shard_idx, shard_path in enumerate(shards):
            x, y, sid = load_shard_tensors(shard_path, args.input_key,
                                           args.target_key,
                                           args.sample_id_key)

            if args.max_rows > 0:
                remain = args.max_rows - max_rows_state["seen_rows_eval"]
                if remain <= 0:
                    break
                take = min(remain, x.shape[0])
                x = x[:take]
                y = y[:take]
                if sid is not None:
                    sid = sid[:take]
                max_rows_state["seen_rows_eval"] += take

            val_mask = split_mask(x.shape[0], sid, args.val_ratio, args.seed,
                                  shard_idx, args.split_by_sample_id)
            val_idx = val_mask.nonzero(as_tuple=False).squeeze(-1)
            if val_idx.numel() == 0:
                continue

            for i in range(0, val_idx.numel(), args.batch_size):
                batch_idx = val_idx[i:i + args.batch_size]
                bx = x.index_select(0, batch_idx).float().to(device)
                by = y.index_select(0, batch_idx).float().to(device)
                if args.input_rms_norm:
                    bx = maybe_rms_norm(bx, args.norm_eps)

                pred = model(bx)
                loss, mse, cos = compute_loss(pred, by, args.mse_weight,
                                              args.cos_weight)
                meter.update(float(loss.detach().cpu()),
                             float(mse.detach().cpu()),
                             float(cos.detach().cpu()),
                             bx.shape[0])

            del x, y, sid

    m = meter.as_dict()
    LOGGER.info(
        "eval epoch=%d rows=%d loss=%.6f mse=%.6f cos=%.6f",
        epoch,
        int(m["rows"]),
        m["loss"],
        m["mse"],
        m["cos"],
    )
    return meter


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    LOGGER.info("device=%s", device)

    shards = list_shards(args.data_dir, args.shard_pattern)
    LOGGER.info("found %d shards", len(shards))

    x0, y0, _ = load_shard_tensors(shards[0], args.input_key, args.target_key,
                                   args.sample_id_key)
    in_dim = int(x0.shape[1])
    out_dim = int(y0.shape[1])
    LOGGER.info("adapter dims: in_dim=%d out_dim=%d", in_dim, out_dim)
    del x0, y0

    model = nn.Linear(in_dim, out_dim, bias=args.bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_limit_state = {"seen_rows_train": 0}
        eval_limit_state = {"seen_rows_eval": 0}

        train_meter = train_epoch(model, optimizer, shards, args, device, epoch,
                                  train_limit_state)
        val_meter = eval_epoch(model, shards, args, device, epoch,
                               eval_limit_state)

        train_metrics = train_meter.as_dict()
        val_metrics = val_meter.as_dict()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_cos": train_metrics["cos"],
            "train_rows": train_metrics["rows"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_cos": val_metrics["cos"],
            "val_rows": val_metrics["rows"],
        }
        history.append(row)

        score = val_metrics["loss"] if val_metrics["rows"] > 0 else train_metrics[
            "loss"]
        if score < best_val:
            best_val = score
            best_state = {
                "weight": model.weight.detach().float().cpu().contiguous(),
                "bias": model.bias.detach().float().cpu().contiguous()
                if model.bias is not None else None,
            }

        LOGGER.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f best=%.6f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            best_val,
        )

    elapsed = max(time.time() - train_start, 1e-6)

    if best_state is None:
        best_state = {
            "weight": model.weight.detach().float().cpu().contiguous(),
            "bias": model.bias.detach().float().cpu().contiguous()
            if model.bias is not None else None,
        }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weight": best_state["weight"],
        "bias": best_state["bias"],
        "meta": {
            "format_version": 1,
            "created_at_unix": int(time.time()),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mse_weight": args.mse_weight,
            "cos_weight": args.cos_weight,
            "val_ratio": args.val_ratio,
            "input_rms_norm": bool(args.input_rms_norm),
            "norm_eps": args.norm_eps,
            "best_score": best_val,
            "elapsed_s": elapsed,
            "source_data_dir": str(args.data_dir),
            "source_pattern": args.shard_pattern,
            "input_key": args.input_key,
            "target_key": args.target_key,
        },
    }
    torch.save(payload, args.output_path)
    LOGGER.info("saved adapter: %s", args.output_path)

    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = args.output_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_obj = {
        "train_history": history,
        "best_score": best_val,
        "elapsed_s": elapsed,
        "output_path": str(args.output_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
    LOGGER.info("saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
