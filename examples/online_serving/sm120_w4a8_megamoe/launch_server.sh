#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "${here}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${here}/.env"
  set +a
fi

if [[ -z ${IMAGE:-} && -f "${here}/IMAGE_TAG" ]]; then
  IMAGE=$(<"${here}/IMAGE_TAG")
fi
: "${IMAGE:?set IMAGE or run build_image.sh}"
: "${MODEL:?set MODEL to the checkpoint directory}"

name=${NAME:-vllm-sm120-w4a8-megamoe}
model_name=${MODEL_NAME:-DeepSeek-V4-Flash-0731}
gpus=${GPUS:-0,1,2,3}
port=${PORT:-8011}
backend=${MOE_BACKEND:-flashinfer_mega_moe}
cache_root=${CACHE_ROOT:-${HOME}/.cache/vllm-sm120-w4a8}
max_num_seqs=${MAX_NUM_SEQS:-28}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
max_model_len=${MAX_MODEL_LEN:-131072}
max_graph=${MAX_CUDAGRAPH_CAPTURE_SIZE:-168}
graph_sizes=${CUDAGRAPH_CAPTURE_SIZES:-${max_graph}}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.88}

case "${backend}" in
  flashinfer_mega_moe|flashinfer_cutlass) ;;
  *) echo "unsupported MOE_BACKEND=${backend}" >&2; exit 2 ;;
esac

IFS=, read -r -a gpu_list <<<"${gpus}"
if [[ ${#gpu_list[@]} -ne 4 ]]; then
  echo "This release requires exactly four same-NUMA GPUs; got GPUS=${gpus}" >&2
  exit 2
fi

mkdir -p "${cache_root}"
docker rm -f "${name}" >/dev/null 2>&1 || true

docker run -d \
  --name "${name}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES="${gpus}" \
  -e NVSHMEM_HEAP_KIND=VIDMEM \
  -e NVSHMEM_REMOTE_TRANSPORT=none \
  -e NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-8G}" \
  -e NCCL_P2P_DISABLE=1 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_MODULE_LOADING=LAZY \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SKIP_DSV4_SPARSE_MLA_WARMUP=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e PYTHONHASHSEED=0 \
  -e OMP_NUM_THREADS=8 \
  -e XDG_CACHE_HOME=/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "${cache_root}:/cache" \
  -v "${MODEL}:/model:ro" \
  --entrypoint bash \
  "${IMAGE}" -lc "
    exec python3 -m vllm.entrypoints.cli.main serve /model \
      --served-model-name '${model_name}' \
      --host 0.0.0.0 --port '${port}' \
      --trust-remote-code --tokenizer-mode deepseek_v4 \
      --tensor-parallel-size 1 --data-parallel-size 4 \
      --enable-expert-parallel --enable-ep-weight-filter \
      --all2all-backend allgather_reducescatter \
      --kernel-config.moe_backend '${backend}' \
      --kernel-config.linear_backend deep_gemm \
      --kv-cache-dtype fp8 --block-size 256 \
      --gpu-memory-utilization '${gpu_memory_utilization}' \
      --max-model-len '${max_model_len}' \
      --max-num-seqs '${max_num_seqs}' \
      --max-num-batched-tokens '${max_num_batched_tokens}' \
      --max-cudagraph-capture-size '${max_graph}' \
      --cudagraph-capture-sizes ${graph_sizes//,/ } \
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}' \
      --async-scheduling --no-enable-prefix-caching
  "

echo "${name} listening on port ${port}"
