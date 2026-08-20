ARG BASE_IMAGE=local/vllm-main-megamoe-runtime:nvshmem370
FROM ${BASE_IMAGE}

ARG VLLM_COMMIT
ARG VLLM_UPSTREAM_COMMIT
ARG FLASHINFER_COMMIT
ARG FLASHINFER_BASE_COMMIT

LABEL org.opencontainers.image.title="vLLM SM120 W4A8 MegaMoE"
LABEL org.opencontainers.image.description="Self-contained vLLM + FlashInfer SM120 W4A8 MegaMoE runtime"
LABEL org.opencontainers.image.revision.vllm="${VLLM_COMMIT}"
LABEL org.opencontainers.image.revision.vllm_upstream="${VLLM_UPSTREAM_COMMIT}"
LABEL org.opencontainers.image.revision.flashinfer="${FLASHINFER_COMMIT}"
LABEL org.opencontainers.image.revision.flashinfer_base="${FLASHINFER_BASE_COMMIT}"

USER root

COPY vllm-overlay/vllm /opt/megamoe-overlay/vllm
COPY flashinfer/moe_ep /opt/megamoe-src/flashinfer/moe_ep
COPY tests /opt/megamoe/tests
COPY delivery /opt/megamoe

RUN set -eux; \
    site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    cp -a /opt/megamoe-overlay/vllm/. "${site_dir}/vllm/"; \
    rm -rf "${site_dir}/flashinfer/moe_ep"; \
    cp -a /opt/megamoe-src/flashinfer/moe_ep "${site_dir}/flashinfer/moe_ep"; \
    find "${site_dir}/vllm" "${site_dir}/flashinfer/moe_ep" \
      -type d -name __pycache__ -prune -exec rm -rf '{}' +; \
    chmod +x /opt/megamoe/*.sh

ENV MEGAMOE_SOURCE_MANIFEST=/opt/megamoe/SOURCE_MANIFEST
WORKDIR /workspace
