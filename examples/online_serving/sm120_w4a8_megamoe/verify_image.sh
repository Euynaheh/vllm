#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image=${IMAGE:-}
if [[ -z ${image} && -f "${here}/IMAGE_TAG" ]]; then
  image=$(<"${here}/IMAGE_TAG")
fi
: "${image:?set IMAGE or run build_image.sh}"

docker run --rm --network none --entrypoint python3 "${image}" \
  /opt/megamoe/smoke_imports.py
