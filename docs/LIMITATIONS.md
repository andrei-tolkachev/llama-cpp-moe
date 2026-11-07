# Validation limits and negative results

The selected profiles are local measured configurations. Short repeated prompts warm the expert cache and do not represent every workload. Current deployment measurements supersede the older installation records; those records are retained under historical results.

- Qwen's accepted frequency/cache-release path measured about 78 tokens/s and about 79–81 GiB memory on the recorded host. A complete full-tool rebuild may vary by a few tokens/s. Repeated starts can change greedy tokens; one ABBA's exact 4096-token match is not universal determinism.
- DeepSeek throughput varies with memory residency and request history. Sustained 25 tokens/s has not been established. CPU/GPU rounding can change greedy tokens; historical distribution probes had mean JS divergence 0.02908 and 6/8 matching top-1.
- Historical numerical checks: 61440 IQ values, 24576 MXFP4 values and 160 CPU/CUDA partitions / 163840 values passed their exact comparisons. This covers those formats/shapes, not all models or races.
- Twelve short tasks are a regression smoke test, not general quality acceptance. Historical paired checks scored Qwen 11/12 and DeepSeek 9/12; current scores are reported separately.
- Capacity 131072 is configured and checked against server properties. Actual Qwen and DeepSeek 32K/64K prefill and three-key retrieval are measured in PREFILL.md; filled 128K/256K and broad long-context quality remain unvalidated. The full inherited optimization campaign was interrupted after 18 samples; exhaustive acceptance was not completed.
- Full inherited experimental flags plus Qwen pinned mappings caused two CUDA failures, one Xid31. The selected Qwen profile excludes that combination. Short successful checks do not establish long-running stability.
- DSpark and Qwen MTP did not establish an accepted gain. KV quantization changed output IDs without an accepted speed win. Selected profiles preserve F16 KV and disable speculative decoding.
- Managed/ordinary host KV, pthread replacement, small CPU pools, Q8 two-row kernels, router lookahead and later scheduler prototypes were not accepted. No unconfirmed accelerations are enabled by default.
- Strata's native engine is faster on the measured short Qwen prompts. Its larger expert cache, resident KV policy, fused graphs, CPU/GPU scheduling and native weight paths differ. 194 HC conversions in its pack may be bypassed by original-GGUF Q8 reads; do not attribute the gap to pack quantization alone. The full scheduler was not ported. See PREFILL.md for the independent async off/on/on/off experiment and its unequal-output limitation.
- Maya's EXL3 DeepSeek weights are not interchangeable with GGUF. Results from different quantizations are not controlled A/B evidence.

Experimental code materially involved AI assistance. This independent repository does not imply upstream review or endorsement.
