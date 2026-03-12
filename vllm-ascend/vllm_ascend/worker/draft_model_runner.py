# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import os
from typing import Any, Dict, List, Optional, Set

import torch

from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput


from vllm.logger import init_logger
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.worker.model_runner_base import (ModelRunnerBase,
                                           ModelRunnerInputBase,
                                           ModelRunnerWrapperBase)

logger = init_logger(__name__)

# A flag to enable debug prints for the updated input tensors
# before each step.
debug_advance_input = False
# A flag to allow GPU advance step for draft model runner.
# Set to False for debugging.
allow_gpu_advance_step = True


class TP1DraftModelRunner(ModelRunnerWrapperBase):
    """Specialized model runner for speculative decoding draft model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    TODOs:
    1. Currently supports only flash-attn, add support for other attn_backends.
    2. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(self, model_runner: ModelRunnerBase):
        super().__init__(model_runner)

        self.indices_of_seq_with_bonus_tokens = None
        self._supports_previous_hidden_states: Optional[bool] = None
        self._warned_previous_hidden_states_unsupported = False

        # Logits alignment controls used by model-alignment experiments.
        self._align_enable = os.getenv("VLLM_ASCEND_DRAFT_ALIGN_ENABLE",
                                       "0") == "1"
        self._align_mode = os.getenv("VLLM_ASCEND_DRAFT_ALIGN_MODE",
                                     "temperature").strip().lower()
        self._align_temperature = self._read_float_env(
            "VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE", 1.0, min_value=1e-6)
        self._align_scale = self._read_float_env("VLLM_ASCEND_DRAFT_ALIGN_SCALE",
                                                 1.0)
        self._align_bias = self._read_float_env("VLLM_ASCEND_DRAFT_ALIGN_BIAS",
                                                0.0)
        self._align_vocab_bias_path = os.getenv(
            "VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH", "").strip()
        self._align_vocab_bias_cpu: Optional[torch.Tensor] = None
        self._align_vocab_bias_by_device: Dict[str, torch.Tensor] = {}
        self._align_config_logged = False
        self._load_vocab_bias_from_env()

        # Draft GPU fast-path capability controls and reason breakdown stats.
        self._gpu_multistep_backend_allowlist = self._read_backend_allowlist_env(
            "VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS",
            "FLASH_ATTN,ASCEND",
        )
        self._gpu_multistep_stats_interval = int(
            os.getenv("VLLM_ASCEND_MS_STATS_INTERVAL", "200"))
        if self._gpu_multistep_stats_interval <= 0:
            self._gpu_multistep_stats_interval = 200

        self.ms_total = 0
        self.ms_hit = 0
        self.ms_fail_disabled = 0
        self.ms_fail_prompt = 0
        self.ms_fail_backend = 0
        self.ms_fail_lora = 0
        self.ms_fail_adapter = 0
        self._gpu_multistep_allowlist_logged = False

    @staticmethod
    def _read_float_env(name: str,
                        default: float,
                        min_value: Optional[float] = None) -> float:
        value = os.getenv(name, str(default))
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r, fallback to %.6f", name, value,
                           default)
            parsed = default
        if min_value is not None and parsed < min_value:
            logger.warning("%s=%.6f is smaller than %.6f, clamped", name,
                           parsed, min_value)
            parsed = min_value
        return parsed

    @staticmethod
    def _read_backend_allowlist_env(name: str, default: str) -> Set[str]:
        raw = os.getenv(name, default)
        allowlist = {
            item.strip().upper() for item in raw.split(",") if item.strip()
        }
        if allowlist:
            return allowlist
        return {
            item.strip().upper() for item in default.split(",")
            if item.strip()
        }

    def _get_attn_backend_name(self) -> str:
        backend = getattr(self, "attn_backend", None)
        get_name = getattr(backend, "get_name", None)
        if callable(get_name):
            try:
                return str(get_name()).upper()
            except Exception:  # pylint: disable=broad-except
                return "<UNKNOWN>"
        return "<UNKNOWN>"

    def _attn_backend_supports_gpu_multi_step(self) -> bool:
        backend = getattr(self, "attn_backend", None)
        supports_fn = getattr(backend, "supports_spec_gpu_multi_step", None)
        if callable(supports_fn):
            try:
                return bool(supports_fn())
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("supports_spec_gpu_multi_step() failed: %s", exc)
        return self._get_attn_backend_name() in self._gpu_multistep_backend_allowlist

    def _attn_metadata_supports_advance_step(self) -> bool:
        backend = getattr(self, "attn_backend", None)
        get_metadata_cls = getattr(backend, "get_metadata_cls", None)
        if not callable(get_metadata_cls):
            return False
        try:
            metadata_cls = get_metadata_cls()
        except Exception:  # pylint: disable=broad-except
            return False
        return hasattr(metadata_cls, "advance_step")

    def _log_gpu_multi_step_stats(self) -> None:
        if not self._gpu_multistep_allowlist_logged:
            logger.info(
                "DraftGPUFastPath allow_backends=%s backend=%s",
                ",".join(sorted(self._gpu_multistep_backend_allowlist)),
                self._get_attn_backend_name(),
            )
            self._gpu_multistep_allowlist_logged = True

        if self.ms_total == 0:
            return
        if self.ms_total % self._gpu_multistep_stats_interval != 0:
            return

        hit_rate = self.ms_hit / max(self.ms_total, 1)
        logger.info(
            "DraftGPUFastPath hit_rate=%.4f total=%d hit=%d fail_disabled=%d "
            "fail_prompt=%d fail_backend=%d fail_lora=%d fail_adapter=%d",
            hit_rate,
            self.ms_total,
            self.ms_hit,
            self.ms_fail_disabled,
            self.ms_fail_prompt,
            self.ms_fail_backend,
            self.ms_fail_lora,
            self.ms_fail_adapter,
        )

    @staticmethod
    def _dense_bias_from_mapping(mapping: Dict[Any, Any]) -> Optional[torch.Tensor]:
        dense: Dict[int, float] = {}
        for key, value in mapping.items():
            if key in ("bias", "vocab_bias", "values"):
                continue
            try:
                idx = int(key)
                val = float(value)
            except (TypeError, ValueError):
                continue
            dense[idx] = val

        if not dense:
            return None
        max_idx = max(dense.keys())
        if max_idx < 0:
            return None
        tensor = torch.zeros(max_idx + 1, dtype=torch.float32)
        for idx, val in dense.items():
            if idx >= 0:
                tensor[idx] = val
        return tensor

    def _load_vocab_bias_from_env(self) -> None:
        path = self._align_vocab_bias_path
        if not path:
            return

        if not os.path.isfile(path):
            logger.warning("Vocab bias file not found: %s", path)
            return

        raw_obj: Any
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    raw_obj = json.load(f)
            else:
                raw_obj = torch.load(path, map_location="cpu")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load vocab bias from %s: %s", path, exc)
            return

        tensor: Optional[torch.Tensor] = None
        if isinstance(raw_obj, torch.Tensor):
            tensor = raw_obj.detach().float().flatten().cpu()
        elif isinstance(raw_obj, list):
            try:
                tensor = torch.tensor(raw_obj, dtype=torch.float32).flatten()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Invalid list vocab bias in %s: %s", path, exc)
        elif isinstance(raw_obj, dict):
            if "bias" in raw_obj:
                try:
                    tensor = torch.tensor(raw_obj["bias"],
                                          dtype=torch.float32).flatten()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Invalid 'bias' field in %s: %s", path, exc)
            elif "vocab_bias" in raw_obj:
                try:
                    tensor = torch.tensor(raw_obj["vocab_bias"],
                                          dtype=torch.float32).flatten()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Invalid 'vocab_bias' field in %s: %s",
                                   path, exc)
            elif "values" in raw_obj:
                try:
                    tensor = torch.tensor(raw_obj["values"],
                                          dtype=torch.float32).flatten()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Invalid 'values' field in %s: %s", path,
                                   exc)
            else:
                tensor = self._dense_bias_from_mapping(raw_obj)

        if tensor is None:
            logger.warning("Unsupported vocab bias payload in %s", path)
            return
        if tensor.numel() == 0:
            logger.warning("Loaded empty vocab bias from %s", path)
            return

        self._align_vocab_bias_cpu = tensor.cpu()
        logger.info("Loaded vocab bias from %s (size=%d)", path, tensor.numel())

    def _model_supports_previous_hidden_states(self) -> bool:
        if self._supports_previous_hidden_states is not None:
            return self._supports_previous_hidden_states

        supports = False
        forward_fn = getattr(self.model, "forward", None)
        if forward_fn is not None:
            try:
                sig = inspect.signature(forward_fn)
                if "previous_hidden_states" in sig.parameters:
                    supports = True
                elif any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values()):
                    supports = True
            except (TypeError, ValueError):
                supports = False

        self._supports_previous_hidden_states = supports
        return supports

    def _get_vocab_bias_for_logits(self,
                                   logits: torch.Tensor) -> Optional[torch.Tensor]:
        if self._align_vocab_bias_cpu is None:
            return None

        vocab_size = logits.shape[-1]
        device_key = f"{logits.device.type}:{logits.device.index}"
        bias = self._align_vocab_bias_by_device.get(device_key)
        if bias is None or bias.dtype != logits.dtype:
            bias = self._align_vocab_bias_cpu.to(device=logits.device,
                                                 dtype=logits.dtype)

        if bias.numel() < vocab_size:
            bias = torch.nn.functional.pad(bias, (0, vocab_size - bias.numel()))
        elif bias.numel() > vocab_size:
            bias = bias[:vocab_size]

        self._align_vocab_bias_by_device[device_key] = bias
        return bias

    def _align_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._align_enable:
            return logits

        mode = self._align_mode
        if mode not in ("temperature", "legacy_affine",
                        "temperature_then_affine"):
            logger.warning("Unknown align mode %r, fallback to temperature",
                           mode)
            mode = "temperature"

        if mode == "temperature":
            logits = logits / self._align_temperature
        elif mode == "legacy_affine":
            logits = logits * self._align_scale + self._align_bias
        else:
            logits = logits / self._align_temperature
            logits = logits * self._align_scale + self._align_bias

        vocab_bias = self._get_vocab_bias_for_logits(logits)
        if vocab_bias is not None:
            logits = logits + vocab_bias

        return logits

    def _maybe_log_align_config(self) -> None:
        if self._align_config_logged:
            return
        self._align_config_logged = True
        logger.info(
            "DRAFT_ALIGN config: enable=%s mode=%s temperature=%.4f scale=%.4f "
            "bias=%.4f vocab_bias_path=%s supports_previous_hidden_states=%s",
            self._align_enable,
            self._align_mode,
            self._align_temperature,
            self._align_scale,
            self._align_bias,
            self._align_vocab_bias_path or "<none>",
            self._model_supports_previous_hidden_states(),
        )

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple

    def _gpu_advance_step(self, model_input: ModelRunnerInputBase,
                          last_output: SamplerOutput) -> ModelRunnerInputBase:
        # Currently, we expect "decode mode" only
        assert not model_input.is_prompt

        # Get num_seqs
        num_seqs = len(model_input.seq_lens)
        num_queries = len(model_input.query_lens)

        # Get output tokens GPU tensor
        sampled_token_ids = last_output.sampled_token_ids
        assert sampled_token_ids is not None

        # Update attn_metadata
        attn_metadata = model_input.attn_metadata
        if not hasattr(attn_metadata, "advance_step"):
            raise RuntimeError(
                "Draft GPU multi-step requires attention metadata.advance_step")

        attn_metadata.advance_step(model_input, sampled_token_ids,
                                   self.block_size, num_seqs, num_queries)

        # Update sampling_metadata
        sampling_metadata = model_input.sampling_metadata
        self._update_sampling_metadata(sampling_metadata, num_seqs,
                                       num_queries)

        # Create new input
        new_model_input = self._model_input_cls(
            input_tokens=model_input.input_tokens,
            input_positions=model_input.input_positions,
            attn_metadata=attn_metadata,
            seq_lens=attn_metadata.seq_lens,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            sampling_metadata=model_input.sampling_metadata,
            is_prompt=False,
        )

        # Ensure we skip CPU samples
        assert new_model_input.sampling_metadata.skip_sampler_cpu_output is True
        # We can reuse sampling tensors since every decode iteration is the same
        new_model_input.sampling_metadata.reuse_sampling_tensors = True

        if debug_advance_input:
            logger.debug("NEW INPUT: ")
            logger.debug("  input_tokens = %s", new_model_input.input_tokens)
            logger.debug("  input_positions = %s",
                         new_model_input.input_positions)
            logger.debug("  seq_lens = %d", new_model_input.seq_lens)
            logger.debug("  query_lens = %d", new_model_input.query_lens)
            logger.debug("  attn_metadata:")
            logger.debug("    seq_lens_tensor: %s",
                         attn_metadata.seq_lens_tensor)
            logger.debug("    slot_mapping: %s", attn_metadata.slot_mapping)
            logger.debug("    block_tables: %s", attn_metadata.block_tables)

        return new_model_input

    def supports_gpu_multi_step(self, execute_model_req: ExecuteModelRequest):
        """Determines if draft_model_runner GPU multi-step can be used.
        Currently required conditions are:
            1. Only decodes
            2. Only flash-attn
            3. No LORA
            4. No prompt_adapter_config
        """
        self.ms_total += 1
        fail_reason = ""

        if not allow_gpu_advance_step:
            self.ms_fail_disabled += 1
            fail_reason = "disabled"

        # We allow multi-step GPU only in decode mode
        if not fail_reason:
            for seq_group in execute_model_req.seq_group_metadata_list:
                if seq_group.is_prompt:
                    self.ms_fail_prompt += 1
                    fail_reason = "prompt"
                    break

        if (not fail_reason and
                (not self._attn_backend_supports_gpu_multi_step()
                 or not self._attn_metadata_supports_advance_step())):
            self.ms_fail_backend += 1
            fail_reason = "backend"

        # TODO: Add support for LORA
        if not fail_reason and self.lora_config:
            self.ms_fail_lora += 1
            fail_reason = "lora"

        # TODO: Add soft-tuning prompt adapter support
        if not fail_reason and self.prompt_adapter_config:
            self.ms_fail_adapter += 1
            fail_reason = "adapter"

        if not fail_reason:
            self.ms_hit += 1

        self._log_gpu_multi_step_stats()
        return not fail_reason

    def set_indices_of_seq_with_bonus_tokens(self,
                                             indices_of_seq_with_bonus_tokens):
        self.indices_of_seq_with_bonus_tokens = indices_of_seq_with_bonus_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelRunnerInputBase,
        kv_caches: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[List[SamplerOutput]]:
        """Executes num_steps forward passes with advacement of input tensors
        on the GPU. Look at supports_gpu_multi_step(..) for pre-conditions.

        Optimizations used:
            1. Input tensors are updated on the GPU directly
            2. Skips GPU=>CPU serialization of sampler outputs (we don't need
                them since we do batch expansion later that uses GPU outputs)
            3. Reuses sampling tensors (since we run only decodes and they have
                a repeating sampling logic)
        """

        # When num_steps == 1, we execute the fallback here for the GPU
        # advance_step, which runs prepare_inputs on CPU and for each spec
        # iteration invokes this function only once
        # (Look at multi-step-worker code)
        is_fallback = num_steps == 1
        supports_previous_hidden_states = \
            self._model_supports_previous_hidden_states()
        self._maybe_log_align_config()
        if not is_fallback:
            # Since we do not broadcast data inside execute_model anymore,
            # we need to figure out the best way to support TP > 1 in this
            # case, because we will at least need to broadcast the sampled
            # tokens to all workers.
            if not self.is_driver_worker:
                raise ValueError("TP1DraftModelRunner only supports TP=1.")

            # Sanity
            if self.lora_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for LORA")
            if self.prompt_adapter_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for "
                                 "prompt_adapter_config")
            if model_input.inputs_embeds is not None:
                raise ValueError("TP1DraftModelRunner has no support for "
                                 "inputs_embeds")
            if model_input.multi_modal_kwargs:
                raise ValueError(
                    "TP1DraftModelRunner has no support for multi_modal_kwargs"
                )
        else:
            if self.lora_config:
                assert model_input.lora_requests is not None
                assert model_input.lora_mapping is not None
                self.set_active_loras(model_input.lora_requests,
                                      model_input.lora_mapping)

            if self.prompt_adapter_config:
                assert model_input.prompt_adapter_requests is not None
                assert model_input.prompt_adapter_mapping is not None
                self.set_active_prompt_adapters(
                    model_input.prompt_adapter_requests,
                    model_input.prompt_adapter_mapping)

            self.attn_state.begin_forward(model_input)

        # Detect exec mode
        assert model_input.attn_metadata is not None
        use_cuda_graph = False
        if model_input.attn_metadata.num_prefills > 0:
            # In this case, execute_model(..) was called directly
            if num_steps > 1:
                raise ValueError(
                    "execute_model(..) of draft_model_runner can be called "
                    "directly only with a single-step prefill")
        else:
            # We can skip CPU samples for spec token generation.
            # (We do allow CPU samples for num_steps == 1 to support the
            # fallback case, where supports_gpu_multi_step(..) does not pass)
            model_input.sampling_metadata.skip_sampler_cpu_output = (
                not is_fallback)

            # CUDA metadata has `use_cuda_graph`; Ascend metadata does not.
            # Default to False on backends without this attribute.
            use_cuda_graph = bool(
                getattr(model_input.attn_metadata, "use_cuda_graph", False))

        # Get model
        if use_cuda_graph:
            if model_input.inputs_embeds is None:
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[model_input.virtual_engine][(
                        graph_batch_size, False)])
            else:
                graph_batch_size = model_input.inputs_embeds.shape[0]
                model_executable = (
                    self.graph_runners[model_input.virtual_engine][(
                        graph_batch_size, True)])

            if previous_hidden_states is not None:
                hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
            else:
                hidden_states = None
        else:
            model_executable = self.model
            hidden_states = previous_hidden_states

        outputs: List[SamplerOutput] = []
        for step in range(num_steps):
            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            model_execute_kwargs: Dict[str, Any] = {}
            if hidden_states is not None and supports_previous_hidden_states:
                model_execute_kwargs["previous_hidden_states"] = hidden_states
            elif hidden_states is not None and \
                    not self._warned_previous_hidden_states_unsupported:
                logger.info("Draft model does not support previous_hidden_states,"
                            " skip forwarding hidden states.")
                self._warned_previous_hidden_states_unsupported = True

            compute_logits_kwargs = {}
            # Run model
            if hasattr(self.model.config, "num_nextn_predict_layers"):
                # for DeepSeek MTP only to use the corresponding layer for
                # each step
                spec_step_idx = kwargs.get("spec_step_idx", step)
                model_execute_kwargs["spec_step_idx"] = spec_step_idx
                compute_logits_kwargs["spec_step_idx"] = spec_step_idx
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config):
                hidden_states = model_executable(
                    input_ids=model_input.input_tokens,
                    inputs_embeds=None,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(
                        multi_modal_kwargs,
                        device=self.device,
                    ),
                    **model_execute_kwargs,
                )

            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata,
                                               **compute_logits_kwargs)
            logits = self._align_logits(logits)
            if not self.is_driver_worker:
                return []
            # Sample the next token.
            output = self.model_runner.sampler(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            outputs.append(output)

            if self.return_hidden_states and is_fallback:
                if use_cuda_graph:
                    indices = model_input.sampling_metadata\
                      .selected_token_indices
                    output.hidden_states = hidden_states[:len(indices)]
                else:
                    output.hidden_states = hidden_states

            if model_input.attn_metadata.num_prefills == 0 \
                and self.indices_of_seq_with_bonus_tokens is not None:
                assert output.sampled_token_ids is not None
                # output.sampled_token_ids should be of shape (num_seqs, 1)
                nums_seqs, num_tokens_per_seq = output.sampled_token_ids.shape
                assert num_tokens_per_seq == 1
                count = 0
                for i in range(nums_seqs):
                    bonus_seq_idx = self.indices_of_seq_with_bonus_tokens[
                        count]
                    if i != bonus_seq_idx:
                        # The following might cause a cpu->gpu sync
                        # However, the performance impact is negligible as we
                        # benchmarked on H100.
                        output.sampled_token_ids[
                            i, :] = model_input.input_tokens[bonus_seq_idx]
                    else:
                        count += 1

            # Prepare inputs for the next step
            if step != num_steps - 1:
                model_input = self._gpu_advance_step(model_input, outputs[-1])

        return outputs











