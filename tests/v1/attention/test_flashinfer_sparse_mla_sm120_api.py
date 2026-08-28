# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as sparse_mod
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_dsv4_empty_prefill_chunk_does_not_call_flashinfer(monkeypatch) -> None:
    class FakeAttention:
        PREFILL_CHUNK_SIZE = 16
        compress_ratio = 1

        @staticmethod
        def _prepare_query(q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            return q

        @staticmethod
        def _as_sparse_cache(cache: torch.Tensor) -> torch.Tensor:
            return cache

    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=0,
        query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
        prefill_swa_indices=torch.empty((0, 1, 4), dtype=torch.int32),
        prefill_swa_lens=torch.empty((0, 1), dtype=torch.int32),
    )

    def fail_if_called(**kwargs) -> None:
        raise AssertionError("FlashInfer must not be called for an empty chunk")

    monkeypatch.setattr(
        sparse_mod,
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
        fail_if_called,
    )
    sparse_mod.DeepseekV4FlashInferSM120Attention._forward_prefill(
        FakeAttention(),
        q=torch.empty((0, 1, 512), dtype=torch.bfloat16),
        compressed_k_cache=None,
        swa_k_cache=torch.empty((1, 1, 1, 512), dtype=torch.bfloat16),
        output=torch.empty((0, 1, 512), dtype=torch.bfloat16),
        attn_metadata=None,
        swa_metadata=metadata,
    )
