# SPDX-License-Identifier: Apache-2.0
"""FP8 paged-MQA logits fallback for consumer Blackwell (SM120).

The vLLM nightly's vendored DeepGEMM has paged-MQA kernels for SM90 and
datacenter Blackwell SM100, but not SM120.  This keeps DeepSeek-V4's decode
indexer on device without flattening or dequantizing the paged FP8 cache.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    kv_scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    next_n,
    context_next_n,
    block_table_width,
    stride_q_row: tl.int64,
    stride_q_h: tl.int64,
    stride_q_d: tl.int64,
    stride_kv_block: tl.int64,
    stride_kv_d: tl.int64,
    stride_scale_block: tl.int64,
    stride_w_row: tl.int64,
    stride_w_h: tl.int64,
    stride_context_batch: tl.int64,
    stride_context_n: tl.int64,
    stride_bt_batch: tl.int64,
    stride_bt_block: tl.int64,
    stride_logits_row: tl.int64,
    stride_logits_k: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // next_n
    slot = row - batch * next_n
    context_slot = tl.minimum(slot, context_next_n - 1)
    context_len = tl.load(
        context_lens_ptr
        + batch * stride_context_batch
        + context_slot * stride_context_n
    )

    h_offsets = tl.arange(0, NUM_HEADS)[:, None]
    d_offsets = tl.arange(0, HEAD_SIZE)
    q = tl.load(
        q_ptr
        + row * stride_q_row
        + h_offsets * stride_q_h
        + d_offsets[None, :] * stride_q_d,
        cache_modifier=".cg",
    )
    head_weights = tl.load(
        weights_ptr + row * stride_w_row + h_offsets * stride_w_h,
        cache_modifier=".cg",
    ).to(tl.float32)

    for tile_start in tl.range(0, context_len, BLOCK_KV):
        logical_offsets = tile_start + tl.arange(0, BLOCK_KV)
        valid = logical_offsets < context_len
        safe_offsets = tl.where(valid, logical_offsets, 0)
        logical_blocks = safe_offsets // CACHE_BLOCK_SIZE
        in_block_offsets = safe_offsets % CACHE_BLOCK_SIZE
        valid &= logical_blocks < block_table_width
        safe_blocks = tl.where(valid, logical_blocks, 0)
        physical_blocks = tl.load(
            block_tables_ptr
            + batch * stride_bt_batch
            + safe_blocks * stride_bt_block,
            mask=valid,
            other=0,
        )

        kv = tl.load(
            kv_ptr
            + physical_blocks[None, :] * stride_kv_block
            + in_block_offsets[None, :] * HEAD_SIZE
            + d_offsets[:, None] * stride_kv_d,
            mask=valid[None, :],
            other=0.0,
        )
        kv_scale = tl.load(
            kv_scale_ptr
            + physical_blocks * stride_scale_block
            + CACHE_BLOCK_SIZE * HEAD_SIZE // 4
            + in_block_offsets * (HEAD_SIZE // 128),
            mask=valid,
            other=0.0,
        )

        scores = tl.dot(q, kv, input_precision="ieee")
        scores *= kv_scale[None, :]
        scores = tl.maximum(scores, 0.0) * head_weights
        scores = tl.sum(scores, axis=0)
        tl.store(
            logits_ptr
            + row * stride_logits_row
            + logical_offsets * stride_logits_k,
            scores,
            mask=valid,
        )


def fp8_paged_mqa_logits_sm120(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute DeepSeek sparse-indexer logits from a paged FP8 cache."""
    assert current_platform.is_device_capability_family(120)
    assert q.dtype == torch.float8_e4m3fn
    assert q.ndim == 4
    assert kv_cache.ndim == 4 and kv_cache.dtype == torch.uint8

    batch_size, next_n, num_heads, head_size = q.shape
    num_blocks, cache_block_size, singleton, cache_width = kv_cache.shape
    assert singleton == 1
    assert cache_width == head_size + 4
    if context_lens.ndim == 1:
        context_lens = context_lens.view(batch_size, 1)
    assert context_lens.ndim == 2
    assert context_lens.shape[0] == batch_size
    assert context_lens.shape[1] in (1, next_n)
    assert weights.shape == (batch_size * next_n, num_heads)
    assert block_tables.shape[0] == batch_size

    q_rows = q.contiguous().view(batch_size * next_n, num_heads, head_size)
    context_lens = context_lens.contiguous()
    block_tables = block_tables.contiguous()
    weights = weights.contiguous()

    assert head_size % 128 == 0
    # CUDA's indexer cache stores all values for a page first, followed by all
    # per-128-element FP32 scales.  Reinterpret the same page storage twice and
    # use explicit offsets in the kernel; the logical 4-D cache view itself is
    # intentionally not token-interleaved.
    kv_values = kv_cache.view(current_platform.fp8_dtype())
    kv_scales = kv_cache.view(torch.float32)

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    _fp8_paged_mqa_logits_kernel[(batch_size * next_n,)](
        q_rows,
        kv_values,
        kv_scales,
        weights,
        context_lens,
        block_tables,
        logits,
        next_n,
        context_lens.shape[1],
        block_tables.shape[1],
        q_rows.stride(0),
        q_rows.stride(1),
        q_rows.stride(2),
        kv_values.stride(0),
        kv_values.stride(3),
        kv_scales.stride(0),
        weights.stride(0),
        weights.stride(1),
        context_lens.stride(0),
        context_lens.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        CACHE_BLOCK_SIZE=cache_block_size,
        BLOCK_KV=64,
        num_warps=4,
        num_stages=2,
    )
    return logits


__all__ = ["fp8_paged_mqa_logits_sm120"]
