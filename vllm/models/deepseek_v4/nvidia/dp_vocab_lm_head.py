# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.distributed import get_dp_group
from vllm.models.deepseek_v4.nvidia.dp_vocab_kernels import (
    fused_bf16_gemm_max,
    fused_bf16_gemm_maxloc,
    pack_maxloc,
    partial_max_storage_shape,
    unpack_maxloc,
)


_MODE_ENV = "VLLM_SM120_DP_VOCAB_LM_HEAD"
_VALID_MODES = {"logits", "greedy"}
_CACHED_IO_ENV = "VLLM_SM120_DP_VOCAB_CACHED_IO"
_FUSED_MAX_ENV = "VLLM_SM120_DP_VOCAB_FUSED_MAX"
_MAXLOC_ENV = "VLLM_SM120_DP_VOCAB_MAXLOC"
_INDEXED_VALIDATE_ENV = "VLLM_SM120_DP_VOCAB_INDEXED_VALIDATE"


@dataclass
class _GreedyBuffers:
    padded_hidden: torch.Tensor
    gathered_hidden: torch.Tensor
    local_logits: torch.Tensor
    local_values: torch.Tensor
    local_indices: torch.Tensor
    global_indices: torch.Tensor
    candidates: torch.Tensor
    gathered_candidates: torch.Tensor
    partial_values: torch.Tensor
    partial_indices: torch.Tensor
    maxloc_keys: torch.Tensor
    maxloc_tokens: torch.Tensor
    valid_rows: int = 0

class DPVocabParallelLMHead:
    """Use a data-parallel group as a temporary vocab-parallel LM head."""

    def __init__(self, mode: str):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"{_MODE_ENV} must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )
        self.mode = mode
        self._debug = os.getenv("VLLM_SM120_DP_VOCAB_DEBUG", "") == "1"
        self._cached_io = os.getenv(_CACHED_IO_ENV, "") == "1"
        self._fused_max = os.getenv(_FUSED_MAX_ENV, "") == "1"
        self._maxloc = os.getenv(_MAXLOC_ENV, "") == "1"
        self._indexed_validate = os.getenv(_INDEXED_VALIDATE_ENV, "") == "1"
        self._greedy_buffers: dict[tuple[object, ...], _GreedyBuffers] = {}
        self._epoch = 0

    @classmethod
    def from_environment(cls) -> "DPVocabParallelLMHead | None":
        mode = os.getenv(_MODE_ENV, "").strip().lower()
        return None if not mode else cls(mode)

    @staticmethod
    def _pad_hidden_states(
        hidden_states: torch.Tensor, bucket_rows: int
    ) -> torch.Tensor:
        num_rows = hidden_states.shape[0]
        if num_rows > bucket_rows:
            raise ValueError(
                f"LM-head rows ({num_rows}) exceed the synchronized DP bucket "
                f"({bucket_rows})"
            )
        if num_rows == bucket_rows:
            return hidden_states
        padding = hidden_states.new_zeros(
            (bucket_rows - num_rows, hidden_states.shape[1])
        )
        return torch.cat((hidden_states, padding), dim=0)

    def _local_logits(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        bucket_rows: int,
    ) -> tuple[torch.Tensor, int, int, int]:
        dp_group = get_dp_group()
        world_size = dp_group.world_size
        if world_size <= 1:
            raise ValueError(f"{_MODE_ENV} requires data_parallel_size > 1")

        weight = lm_head.weight
        if weight.shape[0] % world_size != 0:
            raise ValueError(
                "The padded LM-head vocabulary must be divisible by the DP size: "
                f"{weight.shape[0]} % {world_size} != 0"
            )
        if weight.dtype != hidden_states.dtype:
            raise ValueError(
                f"DP-vocab LM head requires matching BF16/FP16 dtypes, got "
                f"hidden={hidden_states.dtype}, weight={weight.dtype}"
            )

        padded_hidden = self._pad_hidden_states(hidden_states, bucket_rows)
        epoch = self._epoch
        self._epoch += 1
        if self._debug:
            print(
                f"[dp-vocab] epoch={epoch} rank={dp_group.rank_in_group} "
                f"rows={hidden_states.shape[0]} bucket={bucket_rows} enter-hidden-ag",
                flush=True,
            )
        gathered_hidden = dp_group.all_gather(padded_hidden, dim=0)
        if self._debug:
            print(
                f"[dp-vocab] epoch={epoch} rank={dp_group.rank_in_group} "
                "leave-hidden-ag",
                flush=True,
            )

        shard_size = weight.shape[0] // world_size
        vocab_start = dp_group.rank_in_group * shard_size
        vocab_end = vocab_start + shard_size
        bias = None if lm_head.bias is None else lm_head.bias[vocab_start:vocab_end]
        local_logits = F.linear(
            gathered_hidden,
            weight[vocab_start:vocab_end],
            bias,
        )
        return local_logits, vocab_start, shard_size, dp_group.rank_in_group

    def compute_logits(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        bucket_rows: int,
        org_vocab_size: int,
    ) -> torch.Tensor:
        local_logits, _, _, dp_rank = self._local_logits(
            lm_head, hidden_states, bucket_rows
        )
        full_logits = get_dp_group().all_gather(local_logits, dim=-1)
        row_start = dp_rank * bucket_rows
        row_end = row_start + hidden_states.shape[0]
        return full_logits[row_start:row_end, :org_vocab_size]

    def compute_greedy(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        bucket_rows: int,
        org_vocab_size: int,
        source_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._cached_io:
            return self._compute_greedy_cached(
                lm_head,
                hidden_states,
                bucket_rows,
                org_vocab_size,
                source_indices,
            )
        if source_indices is not None:
            hidden_states = torch.index_select(hidden_states, 0, source_indices)
        local_logits, vocab_start, shard_size, dp_rank = self._local_logits(
            lm_head, hidden_states, bucket_rows
        )

        num_valid = max(0, min(shard_size, org_vocab_size - vocab_start))
        if num_valid < shard_size:
            local_logits[:, num_valid:] = -float("inf")
        local_values, local_indices = local_logits.max(dim=-1)
        global_indices = local_indices + vocab_start

        candidates = torch.stack(
            (local_values.float(), global_indices.float()), dim=-1
        )
        if self._debug:
            print(
                f"[dp-vocab] epoch={self._epoch - 1} rank={dp_rank} "
                "enter-candidate-ag",
                flush=True,
            )
        gathered = get_dp_group().all_gather(candidates, dim=-1)
        if self._debug:
            print(
                f"[dp-vocab] epoch={self._epoch - 1} rank={dp_rank} "
                "leave-candidate-ag",
                flush=True,
            )
        gathered = gathered.view(gathered.shape[0], -1, 2)
        winning_rank = gathered[..., 0].argmax(dim=-1, keepdim=True)
        token_ids = gathered[..., 1].gather(1, winning_rank).squeeze(1)

        row_start = dp_rank * bucket_rows
        row_end = row_start + hidden_states.shape[0]
        return token_ids[row_start:row_end].to(torch.int64)

    @staticmethod
    def _all_gather_into(
        dp_group,
        output: torch.Tensor,
        input_: torch.Tensor,
    ) -> None:
        communicator = getattr(dp_group, "device_communicator", None)
        pynccl_comm = getattr(communicator, "pynccl_comm", None)
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_gather(output, input_)
            return
        dist.all_gather_into_tensor(
            output,
            input_,
            group=dp_group.device_group,
        )

    @staticmethod
    def _all_reduce_max_in_place(dp_group, tensor: torch.Tensor) -> None:
        communicator = getattr(dp_group, "device_communicator", None)
        pynccl_comm = getattr(communicator, "pynccl_comm", None)
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_reduce(
                tensor,
                out_tensor=tensor,
                op=dist.ReduceOp.MAX,
            )
            return
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.MAX,
            group=dp_group.device_group,
        )

    def _get_greedy_buffers(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bucket_rows: int,
        world_size: int,
    ) -> _GreedyBuffers:
        shard_size = weight.shape[0] // world_size
        global_rows = bucket_rows * world_size
        key = (
            hidden_states.device,
            hidden_states.dtype,
            hidden_states.shape[1],
            bucket_rows,
            shard_size,
            world_size,
        )
        buffers = self._greedy_buffers.get(key)
        if buffers is not None:
            return buffers
        partial_shape = partial_max_storage_shape(global_rows, shard_size)
        buffers = _GreedyBuffers(
            padded_hidden=hidden_states.new_zeros(
                (bucket_rows, hidden_states.shape[1])
            ),
            gathered_hidden=hidden_states.new_empty(
                (global_rows, hidden_states.shape[1])
            ),
            local_logits=hidden_states.new_empty((global_rows, shard_size)),
            local_values=hidden_states.new_empty(global_rows),
            local_indices=torch.empty(
                global_rows, device=hidden_states.device, dtype=torch.int64
            ),
            global_indices=torch.empty(
                global_rows, device=hidden_states.device, dtype=torch.int64
            ),
            candidates=torch.empty(
                (global_rows, 2),
                device=hidden_states.device,
                dtype=torch.float32,
            ),
            gathered_candidates=torch.empty(
                (world_size * global_rows, 2),
                device=hidden_states.device,
                dtype=torch.float32,
            ),
            partial_values=hidden_states.new_empty(partial_shape),
            partial_indices=torch.empty(
                partial_shape,
                device=hidden_states.device,
                dtype=torch.int32,
            ),
            maxloc_keys=torch.empty(
                global_rows,
                device=hidden_states.device,
                dtype=torch.int64,
            ),
            maxloc_tokens=torch.empty(
                global_rows,
                device=hidden_states.device,
                dtype=torch.int64,
            ),
        )
        self._greedy_buffers[key] = buffers
        return buffers

    def _compute_cached_local_logits(
        self,
        buffers: _GreedyBuffers,
        lm_head,
        weight_shard: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        if lm_head.bias is None:
            torch.mm(
                buffers.gathered_hidden,
                weight_shard.t(),
                out=buffers.local_logits,
            )
        else:
            torch.addmm(
                lm_head.bias[vocab_start:vocab_end],
                buffers.gathered_hidden,
                weight_shard.t(),
                out=buffers.local_logits,
            )

    def _reduce_cached_candidates(
        self,
        buffers: _GreedyBuffers,
        dp_group,
        vocab_start: int,
        shard_size: int,
        org_vocab_size: int,
        bucket_rows: int,
        num_rows: int,
        epoch: int,
        *,
        local_max_ready: bool = False,
        use_maxloc: bool = False,
        maxloc_ready: bool = False,
    ) -> torch.Tensor:
        if not local_max_ready:
            num_valid = max(0, min(shard_size, org_vocab_size - vocab_start))
            if num_valid < shard_size:
                buffers.local_logits[:, num_valid:] = -float("inf")
            torch.max(
                buffers.local_logits,
                dim=-1,
                out=(buffers.local_values, buffers.local_indices),
            )
        row_start = dp_group.rank_in_group * bucket_rows
        row_end = row_start + num_rows
        if use_maxloc:
            if not maxloc_ready:
                pack_maxloc(
                    buffers.local_values,
                    buffers.local_indices,
                    buffers.maxloc_keys,
                    vocab_start,
                )
            self._all_reduce_max_in_place(dp_group, buffers.maxloc_keys)
            unpack_maxloc(buffers.maxloc_keys, buffers.maxloc_tokens)
            return buffers.maxloc_tokens[row_start:row_end]
        buffers.candidates[:, 0].copy_(buffers.local_values)
        torch.add(
            buffers.local_indices,
            vocab_start,
            out=buffers.global_indices,
        )
        buffers.candidates[:, 1].copy_(buffers.global_indices)

        if self._debug:
            print(
                f"[dp-vocab] epoch={epoch} rank={dp_group.rank_in_group} "
                "enter-candidate-ag",
                flush=True,
            )
        self._all_gather_into(
            dp_group, buffers.gathered_candidates, buffers.candidates
        )
        if self._debug:
            print(
                f"[dp-vocab] epoch={epoch} rank={dp_group.rank_in_group} "
                "leave-candidate-ag",
                flush=True,
            )
        merged = buffers.gathered_candidates.view(
            dp_group.world_size, bucket_rows * dp_group.world_size, 2
        ).transpose(0, 1)
        winning_rank = merged[..., 0].argmax(dim=-1, keepdim=True)
        token_ids = merged[..., 1].gather(1, winning_rank).squeeze(1)
        return token_ids[row_start:row_end].to(torch.int64)

    def _compute_greedy_cached(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        bucket_rows: int,
        org_vocab_size: int,
        source_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dp_group = get_dp_group()
        world_size = dp_group.world_size
        if world_size <= 1:
            raise ValueError(f"{_MODE_ENV} requires data_parallel_size > 1")
        weight = lm_head.weight
        if weight.shape[0] % world_size != 0:
            raise ValueError(
                "The padded LM-head vocabulary must be divisible by the DP size: "
                f"{weight.shape[0]} % {world_size} != 0"
            )
        if weight.dtype != hidden_states.dtype:
            raise ValueError(
                f"DP-vocab LM head requires matching BF16/FP16 dtypes, got "
                f"hidden={hidden_states.dtype}, weight={weight.dtype}"
            )

        num_rows = (
            hidden_states.shape[0]
            if source_indices is None
            else source_indices.shape[0]
        )
        if num_rows > bucket_rows:
            raise ValueError(
                f"LM-head rows ({num_rows}) exceed the synchronized DP bucket "
                f"({bucket_rows})"
            )
        buffers = self._get_greedy_buffers(
            hidden_states, weight, bucket_rows, world_size
        )
        if num_rows < buffers.valid_rows:
            buffers.padded_hidden[num_rows : buffers.valid_rows].zero_()
        if source_indices is None:
            buffers.padded_hidden[:num_rows].copy_(hidden_states)
        else:
            torch.index_select(
                hidden_states,
                0,
                source_indices,
                out=buffers.padded_hidden[:num_rows],
            )
            if self._indexed_validate:
                reference_hidden = hidden_states[source_indices]
                mismatches = int(
                    (reference_hidden != buffers.padded_hidden[:num_rows])
                    .sum()
                    .item()
                )
                print(
                    f"[dp-vocab-indexed-validate] epoch={self._epoch} "
                    f"rank={dp_group.rank_in_group} rows={num_rows} "
                    f"mismatches={mismatches}",
                    flush=True,
                )
        buffers.valid_rows = num_rows
        epoch = self._epoch
        self._epoch += 1
        if self._debug:
            print(
                f"[dp-vocab] epoch={epoch} rank={dp_group.rank_in_group} "
                f"rows={num_rows} bucket={bucket_rows} enter-hidden-ag",
                flush=True,
            )
        self._all_gather_into(
            dp_group, buffers.gathered_hidden, buffers.padded_hidden
        )
        if self._debug:
            print(
                f"[dp-vocab] epoch={epoch} rank={dp_group.rank_in_group} "
                "leave-hidden-ag",
                flush=True,
            )

        shard_size = weight.shape[0] // world_size
        vocab_start = dp_group.rank_in_group * shard_size
        vocab_end = vocab_start + shard_size
        weight_shard = weight[vocab_start:vocab_end]
        use_fused_max = (
            self._fused_max
            and buffers.gathered_hidden.shape[0] <= 64
            and buffers.gathered_hidden.shape[1] % 128 == 0
        )
        if use_fused_max:
            num_valid = max(0, min(shard_size, org_vocab_size - vocab_start))
            bias = (
                None
                if lm_head.bias is None
                else lm_head.bias[vocab_start:vocab_end]
            )
            if self._maxloc:
                fused_bf16_gemm_maxloc(
                    buffers.gathered_hidden,
                    weight_shard,
                    bias,
                    num_valid,
                    vocab_start,
                    buffers.partial_values,
                    buffers.partial_indices,
                    buffers.maxloc_keys,
                )
            else:
                fused_bf16_gemm_max(
                    buffers.gathered_hidden,
                    weight_shard,
                    bias,
                    num_valid,
                    buffers.partial_values,
                    buffers.partial_indices,
                    buffers.local_values,
                    buffers.local_indices,
                )
        else:
            self._compute_cached_local_logits(
                buffers,
                lm_head,
                weight_shard,
                vocab_start,
                vocab_end,
            )
        return self._reduce_cached_candidates(
            buffers,
            dp_group,
            vocab_start,
            shard_size,
            org_vocab_size,
            bucket_rows,
            num_rows,
            epoch,
            local_max_ready=use_fused_max,
            use_maxloc=self._maxloc,
            maxloc_ready=use_fused_max and self._maxloc,
        )
