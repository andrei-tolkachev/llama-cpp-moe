# Prefill phase budgets and the remaining output gap

`profiles/qwen-prefill.json` and `profiles/deepseek-prefill.json` add `LLAMA_EXPERIMENTAL_MOE_PHASE_PREFILL_MIB`, batch 4096 and maximum ubatch 4096. The existing `qwen.json`/`deepseek.json` remain the saved small-ubatch decode controls. The opt-in path uses existing llama_context, GGML scheduler and MoE cache lifetimes. It is currently limited to one causal generation slot, ubatch>=512, no embeddings and no allocation-only context with the phase flag.

At construction, SWA/recurrent state is sized for the maximum prefill ubatch. An incoming decode batch above 256 tokens selects the smaller prefill cache and large ubatch; small batches restore ubatch 16 and the full decode cache. A small tail can trigger the transition before the next generated token. GPU work/upload events are completed before scheduler/cache replacement. Model and KV state survive; expert-cache contents and frequency counters are reset. Host page registration is shared across the replacement instead of unregistering/registering all expert pages. The flag must be unset to disable the feature; literal 0 is rejected.

| Qwen input | Earlier small-ubatch input tok/s | Phase input tok/s | Phase input seconds | Immediate answer tok/s |
|---|---:|---:|---:|---:|
|4096|83.36|824.12|4.970|30.31 (40 tokens)|
|32768|78.29|1022.24|32.055|48.38 (512 tokens)|
|65536|not measured|1072.76|61.091|54.52 (512 tokens)|

The 4K input/lookup output and four subsequent 256-token outputs matched the saved small-ubatch run. The 32K input IDs matched but the 512-token outputs did not. All completed Qwen lookup checks in those baseline/candidate runs recovered the three inserted values. Both 32K explanations answered in English despite the requested Russian; lookup scoring does not cover this failure. After 64K, repeated short-output rates were 68.16, 74.59, 76.45, 76.94 tok/s. Capacity remained 131072; F16 KV, no MTP, no prompt cache, no truncation. Qwen phase transitions in the long test took 106–189 ms; one host-registration operation occurred. Peak scope memory was 91.80 GiB with zero scope swap. The earlier ~372 tok/s 4K prototype spent about 7 s re-registering host pages; it is not the selected variant.

DeepSeek's earlier 4K phase test measured 65.35 input tok/s against 13.92 for the old small ubatch. The immediate 28-token lookup answer was 3.86 tok/s, then repeated short requests reached 14.74, 22.32, 24.84, 26.04. The cold expert cache is therefore a real limitation. Current 32K/64K and installed results are recorded separately in `benchmarks/prefill.json` and `docs/RESULTS.md`; do not replace them with the 4K number. No broad quality equivalence is inferred from lookup or 12 short tasks.

DeepSeek 32K completed at **46.03 input tok/s (711.92 s)**, followed by **10.47 tok/s for 512 output tokens**. All three lookup values matched and the explanation used Russian. During a 694.35-second sampling interval inside prefill, `/proc/PID/io` recorded 1088.09 GiB of reads (1.567 GiB/s); sampled GPU utilization had a median of 12%. This supports repeated weight I/O/RAM residency as a major bottleneck for this workload. It is not a claim that software improvements are exhausted. See `benchmarks/deepseek-long-io.json`.

## Reproduce actual 32K/64K

Use the build steps in REPRODUCE.md, one model at a time. The supplied source-code corpus is public MIT-licensed project code; input text is treated as data. Run with a fresh output directory and a scope appropriate to the host:

```bash
systemd-run --user --scope --property=MemoryHigh=106G --property=MemoryMax=108G --property=MemorySwapMax=4G --property=TasksMax=256 -- python3 tools/long_context.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen-prefill.json --out /path/to/new/qwen-phase-long --lengths 32768 65536 --long-answer --post-speed --quality
```

Use `--lengths 32768 65536` (separate option and values) and `deepseek-prefill.json` for DeepSeek. For a control, use `qwen.json`/`deepseek.json` with otherwise identical settings, corpus and lengths. Logs, exact input IDs, output IDs, uncached-token counts, retrieval checks, request times and server timings are saved. Repeat the launches in A/B/B/A order for a stronger comparison. The scripts stop only the server they launched. Final `COMPLETE` means requests completed; inspect `retrieval_passed`, individual quality results and native-EOS lengths separately.

## Why output still trails Strata

The installed Strata 0.1.55 native engine reached 101–125 tok/s on short Qwen inputs from the same original GGUF. Its resident KV 32K inside capacity 128K, expert cache 23.31 GiB and implementation differ. 196 inexact pack conversions are **not proof** that all these errors reach generation: 194 are HC Q8->BF16 and the HC_Q8 native path can read the original GGUF Q8 directly. Quantization alone does not explain the gap.

Our recorded trace had 7.46 ms GPU kernels, 3.49 ms H2D transfers, zero measured H2D/kernel overlap and 2.24–2.60 ms idle time, with about 48 graph launches per token. Strata uses a fuller captured graph, mapped route/activation publication, CPU work concurrent with GPU resident/shared-expert work, and a measured CPU/PCIe dispatch cost. The earlier partial overlap prototype reduced copies but increased idle time and did not establish an accepted speedup. This is unfinished engineering work, not a demonstrated hardware ceiling.

A fresh Strata async off/on/on/off comparison gave 116.99 and 121.94 tok/s warm means, but output IDs differed; no causal 4.2% claim is made. It exceeded 100 even with asynchronous CPU adaptation disabled. Therefore that switch alone cannot explain the whole gap.

The final bounded CUDA experiment fused the small Q8 HC up projection and mix. 60 cases / 78,720 float outputs were exact. For Qwen geometry, the operation fell from 3.569 to 2.985 microseconds. At 96 calls/token, the estimated saving is only 0.056 ms against about 12.7 ms per token (< 0.5%). It was **not integrated**, and no model-throughput improvement was measured. See `benchmarks/hc-up-probe.json`. The saved fast-prefill version remains the selected candidate.
