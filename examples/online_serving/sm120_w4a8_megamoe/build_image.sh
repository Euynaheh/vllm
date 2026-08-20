#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${here}/../../.." && pwd)

vllm_repo=${VLLM_REPO:-${repo_root}}
flashinfer_repo=${FLASHINFER_REPO:-/home/hanyueh/workspace/vllm/flashinfer-sm120-w4a8}
base_image=${BASE_IMAGE:-local/vllm-main-megamoe-runtime:nvshmem370}
expected_base_image_id=sha256:a112afec513c653517942016ea24096c1ff78498140fdfac0e0c6f87a9b47f3d
vllm_upstream=b22afe45ac797ae58e67a7a3ad79ee5714024420
flashinfer_base=c1f9440dacb0498bf06f08daa642831c59a9b97d

vllm_commit=$(git -C "${vllm_repo}" rev-parse HEAD)
flashinfer_commit=$(git -C "${flashinfer_repo}" rev-parse HEAD)
git -C "${vllm_repo}" merge-base --is-ancestor "${vllm_upstream}" HEAD
git -C "${flashinfer_repo}" merge-base --is-ancestor "${flashinfer_base}" HEAD

base_image_id=$(docker image inspect --format '{{.Id}}' "${base_image}")
if [[ ${base_image_id} != "${expected_base_image_id}" \
      && ${ALLOW_UNPINNED_BASE:-0} != 1 ]]; then
  echo "base image ID mismatch: ${base_image_id}" >&2
  echo "expected: ${expected_base_image_id}" >&2
  echo "set ALLOW_UNPINNED_BASE=1 only after revalidating the runtime" >&2
  exit 1
fi

if [[ -n $(git -C "${vllm_repo}" status --porcelain) ]]; then
  echo "vLLM worktree is dirty: ${vllm_repo}" >&2
  exit 1
fi
if [[ -n $(git -C "${flashinfer_repo}" status --porcelain) ]]; then
  echo "FlashInfer worktree is dirty: ${flashinfer_repo}" >&2
  exit 1
fi

tag=${IMAGE:-local/vllm-sm120-w4a8-megamoe:${vllm_commit:0:9}-${flashinfer_commit:0:9}}
context=$(mktemp -d)
trap 'rm -rf "${context}"' EXIT

mkdir -p "${context}/vllm-overlay" "${context}/flashinfer" \
  "${context}/tests/flashinfer" "${context}/delivery"
mapfile -t vllm_files < <(
  git -C "${vllm_repo}" diff --diff-filter=ACMRT --name-only \
    "${vllm_upstream}..${vllm_commit}" -- vllm
)
if [[ ${#vllm_files[@]} -eq 0 ]]; then
  echo "no vLLM overlay files found" >&2
  exit 1
fi
git -C "${vllm_repo}" archive --format=tar "${vllm_commit}" \
  "${vllm_files[@]}" | tar -xf - -C "${context}/vllm-overlay"
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' \
  "${flashinfer_repo}/flashinfer/moe_ep/" "${context}/flashinfer/moe_ep/"
cp "${flashinfer_repo}/tests/moe_ep/test_moe_ep_sm120_w4a8_cutedsl_mega.py" \
  "${context}/tests/flashinfer/"
rsync -a --exclude='Dockerfile' --exclude='IMAGE_TAG' \
  "${here}/" "${context}/delivery/"

cat >"${context}/delivery/SOURCE_MANIFEST" <<EOF
base_image=${base_image}
vllm_upstream_commit=${vllm_upstream}
vllm_commit=${vllm_commit}
flashinfer_pr_4387_head=${flashinfer_base}
flashinfer_commit=${flashinfer_commit}
cutedsl_megamoe_vendor_commit=c603357
EOF

docker build \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "VLLM_COMMIT=${vllm_commit}" \
  --build-arg "VLLM_UPSTREAM_COMMIT=${vllm_upstream}" \
  --build-arg "FLASHINFER_COMMIT=${flashinfer_commit}" \
  --build-arg "FLASHINFER_BASE_COMMIT=${flashinfer_base}" \
  -f "${here}/Dockerfile" \
  -t "${tag}" \
  "${context}"

printf '%s\n' "${tag}" | tee "${here}/IMAGE_TAG"
