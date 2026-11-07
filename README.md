# llama-cpp-moe

Cumulative GGML-compatible patches for large MoE models on limited GPU memory. This independent, AI-assisted project adapts selected Project Maya and Strata mechanisms to an existing llama.cpp fork. It is not an official release of those projects.

On an RTX 5090 / i9-13900K / 128 GiB host, the final installed short-prompt check averaged **78.20 tok/s for Qwen** and **21.23 tok/s for DeepSeek after warmup**. DeepSeek varied from 17.51 to 23.59 tok/s in that check; earlier saved fast-profile means reached about 23.7 tok/s. Sustained 25 tok/s and 100 tok/s Qwen have not been established. The new phase budget switches between a large prefill microbatch and the full decode expert cache without discarding KV state.

Actual uncached Qwen input: **32K in 32.06 s (1022 tok/s)** and **64K in 61.09 s (1073 tok/s)**. The old 32K configuration took 418.56 s (78.29 tok/s) on the same input IDs. These are single starts, not an ABBA; long output IDs differ. Generation immediately after these inputs measured 48.38/54.52 tok/s; the approximately 78 tok/s figure describes a warmed short prompt.

DeepSeek on the actual 32K/64K corpus measured **46.03/50.78 input tok/s** (711.92/1290.61 s), with **10.47/10.54 output tok/s** immediately afterward. These long-request results are separate from the warmed short-prompt figures.

[Русское описание](README_RU.md) · [Build and reproduce](docs/REPRODUCE.md) · [Measurements](docs/RESULTS.md) · [Prefill and output findings](docs/PREFILL.md) · [Context/memory planner](docs/CONTEXT_MEMORY.md) · [Limits](docs/LIMITATIONS.md) · [Attribution](NOTICE.md)

29 sequential patches are supplied against the pinned base in `manifest.json`, with source hashes and an isolated overlay tool. No weights, personal conversations or service configuration are included. Profiles target the measured host and need new memory/affinity tuning elsewhere. Capacity 128K is configured; fully filled 128K/256K have not been validated. F16 KV and non-speculative generation are retained; twelve short tasks and lookup probes do not prove broad quality parity.
