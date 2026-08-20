# SM120 W4A8 MegaMoE Deployment

This directory packages the validated vLLM -> FlashInfer -> SM120 MegaMoE
path into a self-contained image. The server image does not bind-mount or
copy source overlays at startup.

## Pinned Sources

- vLLM upstream base: `b22afe45ac797ae58e67a7a3ad79ee5714024420`
- FlashInfer dependency: PR #4387 head
  `c1f9440dacb0498bf06f08daa642831c59a9b97d`
- Vendored W4A8 kernel source: `bangyus/cutedsl_megamoe`
  `c603357`
- Base runtime image: `local/vllm-main-megamoe-runtime:nvshmem370`

`build_image.sh` records the final vLLM and FlashInfer heads in both image
labels and `/opt/megamoe/SOURCE_MANIFEST`.

## Validated Scope

- Four SM120 GPUs in one NUMA domain.
- Every selected GPU pair supports CUDA peer access and NVSHMEM peer mapping.
- MXFP4 E2M1 weights, MXFP8 E4M3 activations, E8M0 K32 scales, BF16 output.
- Expert parallel size four, tensor parallel size one.
- Public `nvidia-cutlass-dsl==4.6.0`, CUDA 13.3, NVSHMEM 3.7.0,
  `nvshmem4py-cu13==0.3.1`, Python 3.12.

The current FlashInfer vendor drop deliberately exposes only same-NUMA
`p2p_direct`. The standalone cross-NUMA IBGDA transport is not part of this
customer image yet. Do not use eight-rank EP across two NUMA domains with this
release. Eight GPUs are valid only when the framework creates independent,
same-NUMA EP4 groups.

## Build

```bash
cd examples/online_serving/sm120_w4a8_megamoe
VLLM_REPO=/home/hanyueh/workspace/vllm/vllm-megamoe \
FLASHINFER_REPO=/home/hanyueh/workspace/vllm/flashinfer-sm120-w4a8 \
./build_image.sh
```

The script prints the image tag and writes it to `IMAGE_TAG`. Verify the
installed package trees without GPUs:

```bash
./verify_image.sh
```

## Launch

```bash
cp env.example .env
MODEL=/home/hanyueh/workspace/DeepSeek-V4-Flash-0731 \
GPUS=0,1,2,3 \
./launch_server.sh
```

The script launches DP4/EP4 with `flashinfer_mega_moe`. Set
`MOE_BACKEND=flashinfer_cutlass` and a different `PORT`/`NAME` to run the
same attention-parallel configuration with the native MoE backend.

Important variables:

| Variable | Validated value | Meaning |
|---|---|---|
| `NVSHMEM_HEAP_KIND` | `VIDMEM` | Same-NUMA direct peer heap |
| `NVSHMEM_REMOTE_TRANSPORT` | `none` | Reject unsupported remote transport |
| `NVSHMEM_SYMMETRIC_SIZE` | `8G` | Per-process symmetric heap reservation |
| `NCCL_P2P_DISABLE` | `1` | RTX Pro test-platform NCCL IPC workaround |
| `CUTE_DSL_ARCH` | `sm_120a` | Compile the SM120a kernel path |
| `CUDA_MODULE_LOADING` | `LAZY` | Reduce model-startup module pressure |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Select the validated vLLM runner |
| `VLLM_SKIP_DSV4_SPARSE_MLA_WARMUP` | `1` | Skip unsupported mixed sparse-MLA warmup |

`NCCL_P2P_DISABLE=1` affects other NCCL users in the process. It is a launch
policy, not a kernel setting. Revalidate it before removing it on a platform
where CUDA IPC is fixed.

## Tests

Image/import smoke:

```bash
./verify_image.sh
```

FlashInfer kernel correctness and graph replay:

```bash
MODE=single GPUS=0 ./run_correctness.sh
MODE=four GPUS=0,1,2,3 ./run_correctness.sh
```

HTTP server smoke:

```bash
python3 smoke_http.py --url http://127.0.0.1:8011/v1 \
  --model DeepSeek-V4-Flash-0731
```

Strict deterministic backend comparison, after launching MegaMoE and native
servers on different ports:

```bash
python3 compare_backends.py \
  --mega-url http://127.0.0.1:8011/v1 \
  --native-url http://127.0.0.1:8012/v1 \
  --model DeepSeek-V4-Flash-0731
```

The comparison uses greedy decoding and fails on any generated-text mismatch.
For quality evaluation, run the same lm-evaluation-harness revision, task set,
prompts, and seeds against both endpoints; the scripts here are smoke and
regression checks, not a replacement for an accuracy suite.
