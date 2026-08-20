#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image=${IMAGE:-}
if [[ -z ${image} && -f "${here}/IMAGE_TAG" ]]; then
  image=$(<"${here}/IMAGE_TAG")
fi
: "${image:?set IMAGE or run build_image.sh}"

mode=${MODE:-single}
gpus=${GPUS:-0}
case "${mode}" in
  single)
    nproc=1
    selector=single_rank
    extra_env=(-e MEGA_NO_DIST=1)
    ;;
  four)
    nproc=4
    selector=four_rank
    extra_env=()
    ;;
  *) echo "MODE must be single or four" >&2; exit 2 ;;
esac

docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --ulimit memlock=-1:-1 \
  -e CUDA_VISIBLE_DEVICES="${gpus}" \
  -e NVSHMEM_HEAP_KIND=VIDMEM \
  -e NVSHMEM_REMOTE_TRANSPORT=none \
  -e NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-8G}" \
  -e NCCL_P2P_DISABLE=1 \
  -e CUTE_DSL_ARCH=sm_120a \
  "${extra_env[@]}" \
  --entrypoint bash \
  "${image}" -lc "
    python3 -m torch.distributed.run --standalone --nproc_per_node=${nproc} \
      -m pytest -q \
      /opt/megamoe/tests/flashinfer/test_moe_ep_sm120_w4a8_cutedsl_mega.py \
      -k ${selector}
  "
