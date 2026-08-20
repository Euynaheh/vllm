# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, is_deep_gemm_supported


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.  When
    DeepGEMM is disabled, run the same grouped block-FP8 multiplication with
    vLLM's Triton kernel.  This is also needed because a non-DeepGEMM linear
    backend leaves ``wo_a`` in its checkpoint 2-D layout.
    """
    use_deep_gemm = is_deep_gemm_supported()
    if not use_deep_gemm:
        # Triton consumes ordinary FP32 block scales, not the packed TMA
        # layout used by DeepGEMM on Blackwell.
        tma_aligned_scales = False
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if not use_deep_gemm:
        weight_scale = getattr(wo_a, "weight_scale_inv", None)
        if weight_scale is None:
            weight_scale = wo_a.weight_scale
        rows_per_group = wo_a.weight.shape[0] // n_groups
        input_width = heads_per_group * (nope_dim + rope_dim)
        weight = wo_a.weight.view(n_groups, rows_per_group, input_width)
        weight_scale = weight_scale.view(
            n_groups,
            rows_per_group // wo_a.weight_block_size[0],
            input_width // wo_a.weight_block_size[1],
        )
        group_outputs = []
        for group_idx in range(n_groups):
            group_outputs.append(
                w8a8_triton_block_scaled_mm(
                    o_fp8[:, group_idx, :],
                    weight[group_idx],
                    o_scale[:, group_idx, :],
                    weight_scale[group_idx],
                    list(wo_a.weight_block_size),
                    torch.bfloat16,
                )
            )
        z.copy_(torch.stack(group_outputs, dim=1))
        return wo_b(z.flatten(1))

    weight_scale = (
        wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, weight_scale),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
