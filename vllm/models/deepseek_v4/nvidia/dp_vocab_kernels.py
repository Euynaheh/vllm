# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


_BLOCK_N = 64
_BLOCK_K = 128


def _small_m_config(num_rows: int) -> tuple[int, int, int]:
    if num_rows <= 32:
        return 32, 8, 4
    if num_rows <= 64:
        return 64, 4, 4
    raise ValueError("fused BF16 vocab max only supports at most 64 rows")


@triton.jit
def _gemm_partial_max_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    num_valid_n,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        current_k = k_start + offs_k
        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * K + current_k[None, :],
            mask=(offs_m[:, None] < M) & (current_k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + current_k[:, None] + offs_n[None, :] * K,
            mask=(current_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(hidden, weight)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offs_n,
            mask=offs_n < N,
            other=0.0,
        )
        acc += bias[None, :]
    rounded = acc.to(tl.bfloat16)
    rounded = tl.where(offs_n[None, :] < num_valid_n, rounded, -float("inf"))
    max_values = tl.max(rounded, axis=1)
    max_offsets = tl.argmax(rounded, axis=1, tie_break_left=True)
    output_offsets = offs_m * NUM_N_BLOCKS + pid_n
    tl.store(
        partial_values_ptr + output_offsets,
        max_values,
        mask=offs_m < M,
    )
    tl.store(
        partial_indices_ptr + output_offsets,
        pid_n * BLOCK_N + max_offsets,
        mask=offs_m < M,
    )


@triton.jit
def _reduce_partial_max_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_values_ptr,
    output_indices_ptr,
    NUM_N_BLOCKS: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_BLOCK)
    values = tl.load(
        partial_values_ptr + row * NUM_N_BLOCKS + offsets,
        mask=offsets < NUM_N_BLOCKS,
        other=-float("inf"),
    )
    winner = tl.argmax(values, axis=0, tie_break_left=True)
    tl.store(output_values_ptr + row, tl.max(values, axis=0))
    tl.store(
        output_indices_ptr + row,
        tl.load(partial_indices_ptr + row * NUM_N_BLOCKS + winner),
    )


@triton.jit
def _reduce_partial_maxloc_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_ptr,
    vocab_start: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_BLOCK)
    values = tl.load(
        partial_values_ptr + row * NUM_N_BLOCKS + offsets,
        mask=offsets < NUM_N_BLOCKS,
        other=-float("inf"),
    )
    winner = tl.argmax(values, axis=0, tie_break_left=True)
    max_value = tl.max(values, axis=0)
    local_index = tl.load(
        partial_indices_ptr + row * NUM_N_BLOCKS + winner
    )
    bits = tl.cast(max_value.to(tl.float32), tl.uint32, bitcast=True)
    ordered = tl.where(
        (bits & 0x80000000) != 0,
        ~bits,
        bits ^ 0x80000000,
    )
    signed_order = ordered.to(tl.int64) - 0x80000000
    inverse_index = 0xFFFFFFFF - (local_index.to(tl.int64) + vocab_start)
    tl.store(output_ptr + row, signed_order * 0x100000000 + inverse_index)


@triton.jit
def _pack_maxloc_kernel(
    values_ptr,
    indices_ptr,
    output_ptr,
    num_rows: tl.constexpr,
    vocab_start: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.arange(0, block)
    mask = offsets < num_rows
    values = tl.load(values_ptr + offsets, mask=mask, other=-float("inf"))
    bits = tl.cast(values.to(tl.float32), tl.uint32, bitcast=True)
    ordered = tl.where(
        (bits & 0x80000000) != 0,
        ~bits,
        bits ^ 0x80000000,
    )
    signed_order = ordered.to(tl.int64) - 0x80000000
    local_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    global_indices = local_indices.to(tl.int64) + vocab_start
    inverse_index = 0xFFFFFFFF - global_indices
    packed = signed_order * 0x100000000 + inverse_index
    tl.store(output_ptr + offsets, packed, mask=mask)


@triton.jit
def _unpack_maxloc_kernel(
    input_ptr,
    output_ptr,
    num_rows: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.arange(0, block)
    mask = offsets < num_rows
    packed = tl.load(input_ptr + offsets, mask=mask, other=0)
    inverse_index = packed & 0xFFFFFFFF
    tl.store(output_ptr + offsets, 0xFFFFFFFF - inverse_index, mask=mask)


def partial_max_storage_shape(num_rows: int, shard_size: int) -> tuple[int, int]:
    return num_rows, triton.cdiv(shard_size, _BLOCK_N)


def fused_bf16_gemm_max(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_valid_n: int,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_values: torch.Tensor,
    output_indices: torch.Tensor,
) -> None:
    m, k = hidden.shape
    n = weight.shape[0]
    block_m, num_warps, num_stages = _small_m_config(m)
    num_n_blocks = triton.cdiv(n, _BLOCK_N)
    _gemm_partial_max_kernel[(num_n_blocks, triton.cdiv(m, block_m))](
        hidden,
        weight,
        bias,
        partial_values,
        partial_indices,
        num_valid_n,
        M=m,
        N=n,
        K=k,
        NUM_N_BLOCKS=num_n_blocks,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _reduce_partial_max_kernel[(m,)](
        partial_values,
        partial_indices,
        output_values,
        output_indices,
        NUM_N_BLOCKS=num_n_blocks,
        REDUCE_BLOCK=triton.next_power_of_2(num_n_blocks),
        num_warps=4,
    )


def fused_bf16_gemm_maxloc(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_valid_n: int,
    vocab_start: int,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_keys: torch.Tensor,
) -> None:
    m, k = hidden.shape
    n = weight.shape[0]
    block_m, num_warps, num_stages = _small_m_config(m)
    num_n_blocks = triton.cdiv(n, _BLOCK_N)
    _gemm_partial_max_kernel[(num_n_blocks, triton.cdiv(m, block_m))](
        hidden,
        weight,
        bias,
        partial_values,
        partial_indices,
        num_valid_n,
        M=m,
        N=n,
        K=k,
        NUM_N_BLOCKS=num_n_blocks,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _reduce_partial_maxloc_kernel[(m,)](
        partial_values,
        partial_indices,
        output_keys,
        vocab_start=vocab_start,
        NUM_N_BLOCKS=num_n_blocks,
        REDUCE_BLOCK=triton.next_power_of_2(num_n_blocks),
        num_warps=4,
    )


def pack_maxloc(
    values: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    vocab_start: int,
) -> None:
    num_rows = values.shape[0]
    _pack_maxloc_kernel[(1,)](
        values,
        indices,
        output,
        num_rows=num_rows,
        vocab_start=vocab_start,
        block=triton.next_power_of_2(num_rows),
        num_warps=4,
    )


def unpack_maxloc(input_: torch.Tensor, output: torch.Tensor) -> None:
    num_rows = input_.shape[0]
    _unpack_maxloc_kernel[(1,)](
        input_,
        output,
        num_rows=num_rows,
        block=triton.next_power_of_2(num_rows),
        num_warps=4,
    )
