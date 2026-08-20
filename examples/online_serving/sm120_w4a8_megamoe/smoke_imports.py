from __future__ import annotations

from pathlib import Path

import flashinfer
import torch
import vllm

from flashinfer.moe_ep import (
    Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_mega_moe import (
    DeepseekV4FlashInferMegaMoEAdapter,
)


def main() -> None:
    manifest = Path("/opt/megamoe/SOURCE_MANIFEST")
    assert manifest.is_file(), "missing source manifest"
    assert hasattr(torch, "float4_e2m1fn_x2"), "PyTorch lacks packed MXFP4"
    config = Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=4096,
        top_k=6,
    )
    assert config.intermediate_size == 4096
    assert DeepseekV4FlashInferMegaMoEAdapter is not None
    print(f"vllm={vllm.__version__}")
    print(f"flashinfer={flashinfer.__version__}")
    print(manifest.read_text().strip())


if __name__ == "__main__":
    main()
