# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import vllm.models.deepseek_v4.nvidia.flashinfer_mega_moe as adapter_module
from vllm.models.deepseek_v4.nvidia.flashinfer_mega_moe import (
    DeepseekV4FlashInferMegaMoEAdapter,
    is_mega_moe_backend,
)


@dataclass
class _Envelope:
    values: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)


class _FakeLayer(nn.Module):
    def __init__(self, max_tokens: int, hidden_size: int) -> None:
        super().__init__()
        self.output_buffer = torch.empty(
            max_tokens, hidden_size, dtype=torch.bfloat16
        )
        self._staged: _Envelope | None = None

    def stage_inputs(self, tensors: _Envelope) -> None:
        self._staged = tensors

    def compute_staged(self, *, output: torch.Tensor) -> torch.Tensor:
        assert self._staged is not None
        result = self._staged.hidden_states + 1
        output[: result.shape[0]].copy_(result)
        return output[: result.shape[0]]


def test_flashinfer_mega_moe_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("requires torch MXFP4 dtype")

    captured: dict[str, Any] = {}

    def make_layer(**kwargs: Any) -> nn.Module:
        captured.update(kwargs)
        fleet = kwargs["fleet_params"]
        return _FakeLayer(fleet.max_tokens_per_rank, fleet.token_hidden_size)

    api = {
        "BootstrapConfig": _Envelope,
        "FleetParams": _Envelope,
        "MegaConfig": _Envelope,
        "MoEEpLayer": make_layer,
        "MoEEpTensors": _Envelope,
        "PrequantizedMoEWeights": _Envelope,
        "KernelConfig": _Envelope,
    }
    group = SimpleNamespace(world_size=4, rank_in_group=2, device_group=object())
    monkeypatch.setattr(adapter_module, "get_ep_group", lambda: group)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    adapter = DeepseekV4FlashInferMegaMoEAdapter(
        num_experts=8,
        num_local_experts=2,
        top_k=4,
        hidden_size=128,
        intermediate_size=128,
        max_num_tokens=17,
        activation_clamp=7.0,
    )
    monkeypatch.setattr(adapter, "_api", lambda: api)
    monkeypatch.setattr(adapter, "check_runtime_supported", lambda _device: None)

    adapter.finalize_weights(
        torch.zeros(2, 256, 64, dtype=torch.uint8),
        torch.zeros(2, 128, 64, dtype=torch.uint8),
        torch.zeros(2, 256, 4, dtype=torch.uint8),
        torch.zeros(2, 128, 4, dtype=torch.uint8),
    )

    assert adapter.is_finalized
    assert captured["bootstrap"].world_size == 4
    assert captured["bootstrap"].rank == 2
    assert captured["bootstrap"].device == 2
    assert captured["fleet_params"].max_tokens_per_rank == 17
    assert captured["backend"].megakernel.intermediate_size == 128
    assert captured["backend"].megakernel.top_k == 4
    assert captured["weights"].w13.dtype == torch.float4_e2m1fn_x2
    assert captured["weights"].w13_scale.dtype == torch.float8_e8m0fnu

    hidden = torch.zeros(3, 128, dtype=torch.bfloat16)
    output = adapter(
        hidden,
        torch.full((3, 4), 0.25, dtype=torch.float32),
        torch.zeros(3, 4, dtype=torch.int64),
        activation_clamp=7.0,
    )
    torch.testing.assert_close(output, hidden + 1)
    assert output.data_ptr() == adapter.output_buffer.data_ptr()
    assert adapter._fast_tensors is not None
    assert is_mega_moe_backend("flashinfer_mega_moe")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_mega_moe_adapter_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if world_size == 1:
        monkeypatch.setenv("MEGA_NO_DIST", "1")

    group = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        device_group=dist.group.WORLD if world_size > 1 else None,
    )
    monkeypatch.setattr(adapter_module, "get_ep_group", lambda: group)

    hidden = 1024
    intermediate = 1024
    experts = 8
    top_k = 4
    tokens = 67 if world_size > 1 else 64
    local_experts = experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(20260815 + rank)

    def packed_fp4(shape: tuple[int, ...]) -> torch.Tensor:
        low = torch.randint(
            0, 2, shape, dtype=torch.uint8, device="cuda", generator=generator
        )
        high = torch.randint(
            0, 2, shape, dtype=torch.uint8, device="cuda", generator=generator
        )
        return low | (high << 4)

    scale_dtype = torch.float8_e8m0fnu
    w13 = packed_fp4((local_experts, 2 * intermediate, hidden // 2))
    w2 = packed_fp4((local_experts, hidden, intermediate // 2))
    w13_scale = torch.full(
        (local_experts, 2 * intermediate, hidden // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(scale_dtype)
    w2_scale = torch.full(
        (local_experts, hidden, intermediate // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(scale_dtype)

    adapter = DeepseekV4FlashInferMegaMoEAdapter(
        num_experts=experts,
        num_local_experts=local_experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=intermediate,
        max_num_tokens=80,
        activation_clamp=None,
    ).cuda()
    adapter.finalize_weights(w13, w2, w13_scale, w2_scale)
    x = torch.randn(
        tokens,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ) * 0.125
    slots = torch.arange(top_k, device="cuda", dtype=torch.int64)
    topk_ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1) + slots
    ) % experts
    topk_ids[:, 0] = 0
    topk_ids[:, 1] = 1
    topk_weights = torch.full(
        (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )

    try:
        eager0 = adapter(x, topk_weights, topk_ids, activation_clamp=None).clone()
        eager1 = adapter(x, topk_weights, topk_ids, activation_clamp=None).clone()
        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()
        assert torch.isfinite(eager0).all()
        torch.testing.assert_close(eager0, eager1, atol=0.0, rtol=0.0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = adapter(x, topk_weights, topk_ids, activation_clamp=None)
        if world_size > 1:
            dist.barrier()
        graph.replay()
        replay = captured.clone()
        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()
        torch.testing.assert_close(eager0, replay, atol=0.0, rtol=0.0)
    finally:
        if adapter.layer is not None:
            adapter.layer.destroy()
