#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract HS alignment training rows from ShareGPT-style data.

This script builds supervised rows for HS adapter training:
    - input_ids (current token id)
    - previous_hidden_states (teacher hidden at current step)
    - teacher logits (top-k by default)
    - gt_token (next token id)

It can also export the student hidden states at the same step so you can train
an explicit projection from teacher HS space -> student HS space.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("hs_extract_alignment_data")


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class ShareGPTSample:
    sample_id: int
    prompt: str
    completion: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HS-alignment rows from ShareGPT-style dataset.")
    parser.add_argument("--dataset-path",
                        type=Path,
                        required=True,
                        help="Path to ShareGPT JSON.")
    parser.add_argument("--teacher-model",
                        type=str,
                        required=True,
                        help="Teacher/target model path or HF id.")
    parser.add_argument("--student-model",
                        type=str,
                        required=True,
                        help="Student/draft model path or HF id.")
    parser.add_argument("--tokenizer-path",
                        type=str,
                        default="",
                        help="Tokenizer path. Default: teacher model.")
    parser.add_argument("--output-dir",
                        type=Path,
                        required=True,
                        help="Output directory for shards and metadata.")

    parser.add_argument("--num-samples",
                        type=int,
                        default=2000,
                        help="Number of ShareGPT entries to consume.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed for shuffle order.")
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--min-completion-len", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=256)
    parser.add_argument("--max-total-len", type=int, default=2048)
    parser.add_argument("--prepend-bos",
                        action="store_true",
                        help="Prepend tokenizer BOS token if available.")

    parser.add_argument("--teacher-layer",
                        type=int,
                        default=-1,
                        help="Hidden layer index for teacher.")
    parser.add_argument("--student-layer",
                        type=int,
                        default=-1,
                        help="Hidden layer index for student.")

    parser.add_argument("--teacher-device",
                        type=str,
                        default="auto",
                        help="auto|cpu|cuda[:id]|npu[:id]")
    parser.add_argument("--student-device",
                        type=str,
                        default="auto",
                        help="auto|cpu|cuda[:id]|npu[:id]")
    parser.add_argument("--model-dtype",
                        type=str,
                        default="bfloat16",
                        choices=sorted(DTYPE_MAP.keys()))
    parser.add_argument("--storage-dtype",
                        type=str,
                        default="float16",
                        choices=sorted(DTYPE_MAP.keys()))

    parser.add_argument("--teacher-logits-topk",
                        type=int,
                        default=32,
                        help="Top-k logits per row. 0 means disabled.")
    parser.add_argument("--save-full-teacher-logits",
                        action="store_true",
                        help="Store full teacher logits for each row.")

    parser.add_argument("--shard-rows",
                        type=int,
                        default=65536,
                        help="Rows per shard.")
    parser.add_argument("--append",
                        action="store_true",
                        help="Append shards if output dir already has shards.")

    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        default=True)
    parser.add_argument("--no-trust-remote-code",
                        action="store_false",
                        dest="trust_remote_code")
    parser.add_argument("--log-level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    return DTYPE_MAP[dtype_name]


def _parse_device_literal(device_str: str) -> torch.device:
    parts = device_str.split(":")
    if len(parts) == 1:
        return torch.device(parts[0])
    if len(parts) == 2:
        return torch.device(parts[0], int(parts[1]))
    raise ValueError(f"Invalid device string: {device_str}")


def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return _parse_device_literal(device_str)

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


def normalize_model_dtype(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        LOGGER.warning(
            "CPU with %s can be unstable for extraction; fallback to float32.",
            str(dtype))
        return torch.float32
    return dtype


def load_sharegpt_samples(dataset_path: Path,
                          num_samples: int,
                          seed: int) -> List[ShareGPTSample]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{dataset_path} must be a JSON list.")

    samples: List[ShareGPTSample] = []
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue
        conv = entry.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        prompt = conv[0].get("value") if isinstance(conv[0], dict) else None
        completion = conv[1].get("value") if isinstance(conv[1], dict) else None
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        prompt = prompt.strip()
        completion = completion.strip()
        if not prompt or not completion:
            continue
        samples.append(
            ShareGPTSample(sample_id=idx, prompt=prompt, completion=completion))

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:num_samples]


def pick_hidden_layer(hidden_states: Sequence[torch.Tensor],
                      layer_index: int) -> torch.Tensor:
    num_layers = len(hidden_states)
    if layer_index < 0:
        layer_index = num_layers + layer_index
    if layer_index < 0 or layer_index >= num_layers:
        raise IndexError(
            f"layer_index={layer_index} out of range for {num_layers} layers")
    return hidden_states[layer_index]


def safe_tokenize(tokenizer,
                  text: str,
                  *,
                  add_special_tokens: bool = False) -> List[int]:
    out = tokenizer(text, add_special_tokens=add_special_tokens)
    return list(out.input_ids)


def prepare_ids_for_sample(
    tokenizer,
    sample: ShareGPTSample,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    prompt_ids = safe_tokenize(tokenizer,
                               sample.prompt,
                               add_special_tokens=False)
    completion_ids = safe_tokenize(tokenizer,
                                   sample.completion,
                                   add_special_tokens=False)
    if len(prompt_ids) < args.min_prompt_len:
        return None
    if len(completion_ids) < args.min_completion_len:
        return None
    if len(prompt_ids) > args.max_prompt_len:
        return None

    completion_ids = completion_ids[:args.max_completion_len]

    prefix: List[int] = []
    if args.prepend_bos and tokenizer.bos_token_id is not None:
        prefix.append(int(tokenizer.bos_token_id))
    prompt_start = len(prefix)
    full_ids = prefix + prompt_ids + completion_ids
    prompt_end = prompt_start + len(prompt_ids)
    if len(full_ids) > args.max_total_len:
        cut = args.max_total_len - prompt_end
        if cut <= 0:
            return None
        completion_ids = completion_ids[:cut]
        full_ids = prefix + prompt_ids + completion_ids

    if not completion_ids:
        return None
    if prompt_end <= 0:
        return None

    return {
        "full_ids": full_ids,
        "prompt_end": prompt_end,
        "completion_len": len(completion_ids),
    }


def run_forward(
    model: AutoModelForCausalLM,
    ids: List[int],
    device: torch.device,
    hidden_layer: int,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden = pick_hidden_layer(out.hidden_states, hidden_layer).squeeze(0)
    logits = out.logits.squeeze(0)
    return {"hidden": hidden, "logits": logits}


class ShardWriter:

    def __init__(self, output_dir: Path, shard_rows: int,
                 initial_shard_index: int):
        self.output_dir = output_dir
        self.shard_rows = shard_rows
        self.shard_index = initial_shard_index
        self._buffer: Dict[str, List[torch.Tensor]] = {}
        self._rows = 0
        self.total_rows = 0
        self.total_shards = 0

    def add(self, batch: Dict[str, torch.Tensor]) -> None:
        rows = int(batch["gt_token_ids"].shape[0])
        if rows <= 0:
            return
        for key, value in batch.items():
            self._buffer.setdefault(key, []).append(value.detach().cpu())
        self._rows += rows
        self.total_rows += rows
        if self._rows >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if self._rows == 0:
            return
        payload_tensors = {
            key: torch.cat(parts, dim=0)
            for key, parts in self._buffer.items()
        }
        shard_path = self.output_dir / f"hs_align_shard_{self.shard_index:05d}.pt"
        torch.save(
            {
                "format_version": 1,
                "rows": self._rows,
                "tensors": payload_tensors,
            },
            shard_path,
        )
        LOGGER.info("Wrote shard %s rows=%d", shard_path.name, self._rows)
        self._buffer.clear()
        self._rows = 0
        self.shard_index += 1
        self.total_shards += 1


def find_initial_shard_index(output_dir: Path, append: bool) -> int:
    existing = sorted(output_dir.glob("hs_align_shard_*.pt"))
    if not existing:
        return 0
    if not append:
        raise FileExistsError(
            f"{output_dir} already contains shards, pass --append to continue.")
    last_name = existing[-1].stem
    suffix = last_name.split("_")[-1]
    return int(suffix) + 1


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_shard_index = find_initial_shard_index(args.output_dir, args.append)

    tokenizer_path = args.tokenizer_path or args.teacher_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    teacher_device = resolve_device(args.teacher_device)
    student_device = resolve_device(args.student_device)
    model_dtype = normalize_model_dtype(parse_torch_dtype(args.model_dtype),
                                        teacher_device)
    student_dtype = normalize_model_dtype(parse_torch_dtype(args.model_dtype),
                                          student_device)
    storage_dtype = parse_torch_dtype(args.storage_dtype)

    LOGGER.info("teacher_device=%s student_device=%s", teacher_device,
                student_device)
    LOGGER.info("teacher_model=%s student_model=%s", args.teacher_model,
                args.student_model)
    LOGGER.info("model_dtype teacher=%s student=%s storage=%s", model_dtype,
                student_dtype, storage_dtype)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=student_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_model.to(teacher_device).eval()
    student_model.to(student_device).eval()

    samples = load_sharegpt_samples(args.dataset_path, args.num_samples,
                                    args.seed)
    LOGGER.info("Loaded %d candidate ShareGPT samples", len(samples))

    writer = ShardWriter(args.output_dir, args.shard_rows, start_shard_index)
    meta_path = args.output_dir / "sample_meta.jsonl"
    meta_f = meta_path.open("a" if args.append else "w", encoding="utf-8")

    processed_samples = 0
    skipped_samples = 0
    extracted_steps = 0
    start_ts = time.time()

    for local_idx, sample in enumerate(samples):
        packed = prepare_ids_for_sample(tokenizer, sample, args)
        if packed is None:
            skipped_samples += 1
            continue

        full_ids: List[int] = packed["full_ids"]
        prompt_end: int = packed["prompt_end"]
        completion_len: int = packed["completion_len"]

        teacher_out = run_forward(teacher_model, full_ids, teacher_device,
                                  args.teacher_layer)
        student_out = run_forward(student_model, full_ids, student_device,
                                  args.student_layer)

        teacher_hidden = teacher_out["hidden"]
        teacher_logits = teacher_out["logits"]
        student_hidden = student_out["hidden"]

        ids_tensor = torch.tensor(full_ids, dtype=torch.long)
        pos_start = prompt_end - 1
        pos_end = len(full_ids) - 2
        if pos_end < pos_start:
            skipped_samples += 1
            continue
        positions = torch.arange(pos_start, pos_end + 1, dtype=torch.long)

        teacher_prev_hs = teacher_hidden.index_select(
            0, positions.to(teacher_hidden.device))
        student_prev_hs = student_hidden.index_select(
            0, positions.to(student_hidden.device))
        current_token_ids = ids_tensor.index_select(0, positions)
        gt_token_ids = ids_tensor.index_select(0, positions + 1)

        teacher_prev_hs = teacher_prev_hs.to(dtype=storage_dtype)
        student_prev_hs = student_prev_hs.to(dtype=storage_dtype)

        batch: Dict[str, torch.Tensor] = {
            "sample_row_id":
            torch.full((positions.numel(), ),
                       sample.sample_id,
                       dtype=torch.int64),
            "decode_pos":
            positions.clone(),
            "input_token_ids":
            current_token_ids.clone(),
            "gt_token_ids":
            gt_token_ids.clone(),
            "previous_hidden_states":
            teacher_prev_hs,
            "student_hidden_states":
            student_prev_hs,
        }

        if args.teacher_logits_topk > 0:
            selected_logits = teacher_logits.index_select(
                0, positions.to(teacher_logits.device))
            k = min(args.teacher_logits_topk, selected_logits.shape[-1])
            topk_values, topk_ids = torch.topk(selected_logits, k=k, dim=-1)
            topk_logprobs = torch.log_softmax(selected_logits.float(),
                                              dim=-1).gather(-1, topk_ids)
            batch["teacher_topk_ids"] = topk_ids.to(dtype=torch.int32)
            batch["teacher_topk_logits"] = topk_values.to(dtype=storage_dtype)
            batch["teacher_topk_logprobs"] = topk_logprobs.to(
                dtype=storage_dtype)

        if args.save_full_teacher_logits:
            selected_logits = teacher_logits.index_select(
                0, positions.to(teacher_logits.device))
            batch["teacher_full_logits"] = selected_logits.to(dtype=storage_dtype)

        writer.add(batch)

        meta_f.write(
            json.dumps(
                {
                    "local_index": local_idx,
                    "sample_id": sample.sample_id,
                    "prompt_chars": len(sample.prompt),
                    "completion_chars": len(sample.completion),
                    "prompt_end": prompt_end,
                    "completion_len": completion_len,
                    "steps": int(positions.numel()),
                },
                ensure_ascii=False,
            ) + "\n")

        processed_samples += 1
        extracted_steps += int(positions.numel())
        if processed_samples % 50 == 0:
            elapsed = max(time.time() - start_ts, 1e-6)
            LOGGER.info(
                "progress samples=%d steps=%d skipped=%d speed=%.2f rows/s",
                processed_samples,
                extracted_steps,
                skipped_samples,
                extracted_steps / elapsed,
            )

    writer.flush()
    meta_f.close()

    elapsed = max(time.time() - start_ts, 1e-6)
    manifest = {
        "format_version": 1,
        "created_at_unix": int(time.time()),
        "dataset_path": str(args.dataset_path),
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "tokenizer_path": tokenizer_path,
        "num_requested_samples": args.num_samples,
        "processed_samples": processed_samples,
        "skipped_samples": skipped_samples,
        "extracted_rows": writer.total_rows,
        "total_shards_written": writer.total_shards,
        "teacher_layer": args.teacher_layer,
        "student_layer": args.student_layer,
        "teacher_device": str(teacher_device),
        "student_device": str(student_device),
        "model_dtype": args.model_dtype,
        "storage_dtype": args.storage_dtype,
        "teacher_logits_topk": args.teacher_logits_topk,
        "save_full_teacher_logits": bool(args.save_full_teacher_logits),
        "rows_per_second": writer.total_rows / elapsed,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    LOGGER.info("Done. rows=%d shards=%d elapsed=%.1fs",
                writer.total_rows,
                writer.total_shards,
                elapsed)
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
