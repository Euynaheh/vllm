"""Opt-in CUDA-event profiler for the SM120 MegaMoE full-model adapter.

This module is diagnostic-only.  It installs no hooks unless
``SM120_PREFILL_PROFILE_TRIGGER`` is set.  One marker value profiles one model
forward and writes one JSON artifact per rank.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


_STATE: dict[str, Any] = {
    "active": False,
    "last_marker": None,
    "events": [],
    "names": {},
    "current_layer": None,
    "flashinfer_hooks": False,
}


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _rows(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 0


@contextlib.contextmanager
def _range(kind: str, name: str):
    if not _STATE["active"]:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    cpu_start = time.perf_counter_ns()
    try:
        yield
    finally:
        cpu_us = (time.perf_counter_ns() - cpu_start) / 1000.0
        end.record()
        _STATE["events"].append(
            {
                "kind": kind,
                "name": name,
                "layer": _STATE["current_layer"],
                "start": start,
                "end": end,
                "cpu_us": cpu_us,
            }
        )


def _wrap_method(cls: type, method_name: str, kind: str, name: str) -> None:
    marker = f"_sm120_prefill_profile_{method_name}"
    if getattr(cls, marker, False):
        return
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        with _range(kind, name):
            return original(self, *args, **kwargs)

    setattr(cls, method_name, wrapped)
    setattr(cls, marker, True)


def _install_flashinfer_hooks() -> None:
    if _STATE["flashinfer_hooks"]:
        return
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cutedsl.backend import (
        Sm120Mxfp4Mxfp8CutedslMegaKernelBackend,
    )
    from flashinfer.moe_ep.kernel_src.sm120.split_cutedsl_megakernel.shim.runtime import (
        MegaMoESm120W4A8Frontend,
    )
    from moe_sm120_mxfp8_split.runtime.green_context import NativeGreenContextGraph

    _wrap_method(
        Sm120Mxfp4Mxfp8CutedslMegaKernelBackend,
        "stage_inputs",
        "adapter_phase",
        "stage_inputs",
    )
    _wrap_method(
        Sm120Mxfp4Mxfp8CutedslMegaKernelBackend,
        "compute",
        "adapter_phase",
        "compute_total",
    )
    _wrap_method(
        MegaMoESm120W4A8Frontend,
        "_reset_execution",
        "adapter_phase",
        "reset_workspace",
    )
    _wrap_method(
        NativeGreenContextGraph,
        "launch",
        "adapter_phase",
        "green_child_graph",
    )
    _STATE["flashinfer_hooks"] = True


def _module_name(module: Any, fallback: str) -> str:
    return str(_STATE["names"].get(id(module), getattr(module, "prefix", fallback)))


def _wrap_layer_class(cls: type, kind: str) -> None:
    marker = f"_sm120_prefill_profile_{kind}"
    if getattr(cls, marker, False):
        return
    original = cls.forward

    def wrapped(self, *args, **kwargs):
        name = _module_name(self, cls.__name__)
        old_layer = _STATE["current_layer"]
        if kind in ("layer", "moe"):
            _STATE["current_layer"] = name
        try:
            with _range(kind, name):
                return original(self, *args, **kwargs)
        finally:
            _STATE["current_layer"] = old_layer

    cls.forward = wrapped
    setattr(cls, marker, True)


def _event_records(model_start: torch.cuda.Event) -> list[dict[str, Any]]:
    records = []
    for item in _STATE["events"]:
        records.append(
            {
                "kind": item["kind"],
                "name": item["name"],
                "layer": item["layer"],
                "start_us": model_start.elapsed_time(item["start"]) * 1000.0,
                "duration_us": item["start"].elapsed_time(item["end"]) * 1000.0,
                "cpu_us": item["cpu_us"],
            }
        )
    return records


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("layer", "attention", "moe", "adapter_phase"):
        selected = [record for record in records if record["kind"] == kind]
        values = [record["duration_us"] for record in selected]
        result[kind] = {
            "count": len(values),
            "sum_us": sum(values),
            "mean_us": sum(values) / len(values) if values else 0.0,
            "min_us": min(values) if values else 0.0,
            "max_us": max(values) if values else 0.0,
            "cpu_sum_us": sum(record["cpu_us"] for record in selected),
            "cpu_mean_us": (
                sum(record["cpu_us"] for record in selected) / len(selected)
                if selected
                else 0.0
            ),
        }
    phases: dict[str, Any] = {}
    for name in sorted(
        {record["name"] for record in records if record["kind"] == "adapter_phase"}
    ):
        values = [
            record["duration_us"]
            for record in records
            if record["kind"] == "adapter_phase" and record["name"] == name
        ]
        phases[name] = {
            "count": len(values),
            "sum_us": sum(values),
            "mean_us": sum(values) / len(values),
            "min_us": min(values),
            "max_us": max(values),
            "cpu_sum_us": sum(
                record["cpu_us"]
                for record in records
                if record["kind"] == "adapter_phase" and record["name"] == name
            ),
        }
    result["adapter_phases"] = phases
    return result


def install(model_globals: dict[str, Any]) -> None:
    trigger = os.getenv("SM120_PREFILL_PROFILE_TRIGGER")
    if not trigger:
        return

    decoder_cls = model_globals["DeepseekV4DecoderLayer"]
    moe_cls = model_globals["DeepseekV4MoE"]
    causal_cls = model_globals["DeepseekV4ForCausalLM"]
    _wrap_layer_class(decoder_cls, "layer")
    _wrap_layer_class(moe_cls, "moe")

    original = causal_cls.forward

    def profiled_forward(module, *args, **kwargs):
        marker_path = Path(trigger)
        try:
            marker_value = marker_path.read_text().strip()
        except OSError:
            marker_value = ""
        rows = _rows(args, kwargs)
        requested_rows = int(os.getenv("SM120_PREFILL_PROFILE_ROWS", "0"))
        should_profile = (
            bool(marker_value)
            and marker_value != _STATE["last_marker"]
            and not _STATE["active"]
            and (requested_rows == 0 or rows == requested_rows)
        )
        if not should_profile:
            return original(module, *args, **kwargs)

        _install_flashinfer_hooks()
        names: dict[int, str] = {}
        for index, layer in enumerate(module.model.layers):
            layer_name = f"model.layers.{index}"
            names[id(layer)] = layer_name
            if hasattr(layer, "attn"):
                names[id(layer.attn)] = f"{layer_name}.attn"
                _wrap_layer_class(type(layer.attn), "attention")
            if hasattr(layer, "ffn"):
                names[id(layer.ffn)] = f"{layer_name}.ffn"
        _STATE["names"] = names
        _STATE["events"] = []
        _STATE["last_marker"] = marker_value
        _STATE["active"] = True
        model_start = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)
        model_start.record()
        cpu_start = time.perf_counter_ns()
        try:
            result = original(module, *args, **kwargs)
            model_end.record()
            torch.cuda.synchronize()
        finally:
            _STATE["active"] = False
        cpu_us = (time.perf_counter_ns() - cpu_start) / 1000.0
        records = _event_records(model_start)
        payload = {
            "marker": marker_value,
            "rank": _rank(),
            "rows": rows,
            "model_gpu_us": model_start.elapsed_time(model_end) * 1000.0,
            "model_cpu_us_including_final_sync": cpu_us,
            "summary": _summary(records),
            "records": records,
        }
        destination = Path(
            os.getenv("SM120_PREFILL_PROFILE_DIR", "/benchmark-results/prefill-layer")
        )
        destination.mkdir(parents=True, exist_ok=True)
        safe_marker = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in marker_value
        )
        output = destination / f"{safe_marker}-rank{payload['rank']}.json"
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[SM120 prefill profile] wrote {output}", flush=True)
        return result

    causal_cls.forward = profiled_forward


__all__ = ["install"]
