# Selected prefill build: installed verification

The selected build contains 29 cumulative patches and the opt-in phase budgets. Both installed model entry points were checked after activation: executable/shared-library locations, exact profile environment, single slot, capacity 131072 and HTTP web UI. The preceding source, profiles and complete build remain available for rollback; no application core or services were changed.

| Model | First short request tok/s | Next three short requests tok/s | Warm mean | Short tasks |
|---|---:|---|---:|---:|
| qwen | 63.83 | 77.75 / 78.37 / 78.48 | 78.20 | 11/12 |
| deepseek | 15.95 | 22.60 / 17.51 / 23.59 | 21.23 | 9/12 |

These short repeated technical prompts are different from long-document generation. The source-identical experimental build completed these uncached long inputs at capacity 131072:

| Model | Actual input tokens | Input tok/s | Input seconds | Immediate answer tok/s | Answer tokens | Three-key lookup |
|---|---:|---:|---:|---:|---:|---|
| qwen | 32768 | 1022.24 | 32.06 | 48.38 | 512 | PASS |
| qwen | 65536 | 1072.76 | 61.09 | 54.52 | 512 | PASS |
| deepseek | 32768 | 46.03 | 711.92 | 10.47 | 512 | PASS |
| deepseek | 65536 | 50.78 | 1290.61 | 10.54 | 512 | PASS |

The clean full production build separately passed 4K uncached retrieval and completed the same 12 short tasks through the public long-context runner, scoring 11/12 for Qwen and 9/12 for DeepSeek. That Qwen preflight had short rates 61.30, 77.88, 64.73, 78.80 tok/s (warm mean 73.80); the third response had different output IDs. The DeepSeek preflight had 15.64, 23.40, 23.60, 22.07 (warm mean 23.02). These runs remain in the data, including the slower Qwen response. The portable screen wrapper and startup planner were also run on the real Qwen model. Raw numeric summaries and binary hashes are in `benchmarks/selected-build.json`; prefill comparisons are in `benchmarks/prefill.json`. General quality parity, full filled 128K/256K and long-running stability are not established. Qwen 32K explanations used English despite the Russian request; long outputs differ from baseline. The three-key score does not measure complete instruction following.

The final Q8 HC CUDA experiment was not integrated: its measured micro-operation savings imply less than 0.5% token-time headroom, and no model-throughput gain was measured. Details and source are supplied in PREFILL.md and experiments/.

## Previous accepted decode build and historical controls


RTX 5090 / i9-13900K / 128 GiB. One request at a time, capacity 131072, F16 KV, no MTP/DSpark, greedy seed 42, short technical prompt (Qwen 154 tokens; DeepSeek 161), up to 256 output tokens with native EOS. These are request-level decode rates, not end-to-end latency or guarantees.

| Series | First request tok/s | Subsequent requests tok/s | Warm mean | Peak RAM GiB | Peak swap MiB |
|---|---:|---|---:|---:|---:|
| qwen-cache-only-preflight | 62.87 | 76.59 / 77.11 / 76.67 | 76.79 | 79.13 | 0.0 |
| qwen-installed | 62.92 | 77.10 / 77.02 / 77.34 | 77.15 | 90.35 | 177.5 |
| deepseek-cache-only-preflight | 13.20 | 15.55 / 22.05 / 18.11 | 18.57 | 106.00 | 695.6 |
| deepseek-cache-only-old-control | 14.75 | 19.14 / 19.65 / 21.46 | 20.08 | 106.00 | 686.6 |
| deepseek-installed | 16.95 | 19.07 / 23.37 / 23.56 | 22.00 | 106.01 | 866.6 |
| deepseek-restore-frozen-fast | 13.96 | 19.90 / 27.21 / 24.13 | 23.75 | 106.00 | 733.8 |
| deepseek-clean-fast | 15.42 | 21.97 / 22.23 / 26.95 | 23.72 | 106.00 | 706.4 |

The installed checks use the actual production launcher and verify executable/library locations, profile flags, one slot, context capacity and HTTP web UI. The previous build remains available for rollback. DeepSeek uses the preserved frequency-cache and fused CPU gate/up/clamp profile. Cache-only rows are earlier controls, not the selected configuration. The replay of the preserved fast build is included separately from the new clean build and the installed check. MemoryHigh causes reclaim in DeepSeek; changing residency and output routing limit interpretation of small speed differences.

DeepSeek preflight vs previous-build output identity per request: [True, True, False, False]. This is not a general quality comparison.

## Short objective tasks

- qwen-installed: 11/12; strict JSON/value failures: python.
- deepseek-installed: 9/12; strict JSON/value failures: python, units, dependence.

These twelve fixed tasks are only a regression smoke test. Historical paired baseline/candidate scores were Qwen 11/12 and DeepSeek 9/12. No broad quality parity is claimed. The separate 256K allocation/fill experiment is described in CONTEXT_MEMORY.md; a configured capacity alone is not a filled-context test.

## Accepted cache-policy experiment

The frozen ABBA measured 74.2554 -> 78.5694 tokens/s (+5.8097%). All 4096 recorded output tokens matched. Full per-request timings and hashes are in cache-abba.json. New full-tool builds are reported above separately, without replacing their measurements by the frozen result.

The older benchmarks/results.json describes earlier experimental and installation states; its roughly 60 tokens/s Qwen series is superseded by the accepted profile above. Later scheduler prototypes were not selected. Strata native-pack and Maya measurements remain historical comparisons with the arithmetic/quantization limits in LIMITATIONS.md.

## Reproduction status

The cumulative overlay is replayed from the pinned Git base and compared to the built source. Unit tests exercise tooling and fixtures; the frequency test checks the actual extracted cache structure. The CPU gate/up test passed 1392 cases and 5,554,176 exact float comparisons, including final FFN output. The portable screen.py wrapper is separately labelled in PACKAGE_CHECKS.json; historical measurements and installed checks use their recorded host runners.
