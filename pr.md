# Make the softplus/sqrt top-k reference deterministic for ties

## Changes

- Uses a stable descending sort in the shared PyTorch reference so equal scores resolve to lower expert IDs, matching the CUDA/ROCm and DeepSeek-V4 kernels.
- Leaves hash routing unchanged: token-to-expert lookup-table entries continue to define the selected IDs and their order.
- Makes top-k cutoff ties an explicit regression case by setting the first real gating row to all zeros instead of relying on accidental low-precision collisions.
- Keeps the padding test on the shared exact-ID and weight oracle while preserving its explicit padding and NaN assertions.

## Why

The padding test can fail when BF16 routing scores tie at the top-k cutoff because `torch.topk` does not guarantee which tied indices it returns, while the vLLM kernels deterministically prefer lower expert IDs. This exact one-expert mismatch appeared in [AMD CI #11834, MI355 MoE Test 3](https://buildkite.com/vllm/amd-ci/builds/11834#019fdfcf-a69f-439e-a7bf-a35ce8955218), [AMD CI #11845, MI355 MoE Test 2](https://buildkite.com/vllm/amd-ci/builds/11845#019fe374-ac4a-4f92-bad5-9620658cf9a5), and [AMD CI #11862, MI355 MoE Test 3](https://buildkite.com/vllm/amd-ci/builds/11862#019fe4e5-8e16-4f8e-b421-bd548b93dcf5), all with 384 BF16 experts and the same 115-ID delta. The latter two were Andreas's runs, and #11862 failed shortly before this fix was authored, making them likely motivators even though this branch has no Buildkite run of its own. A stable reference removes the false failure for every caller of the helper and preserves exact coverage of the kernels' intentional tie-break, routing weights, hash lookup, and padding behavior.
