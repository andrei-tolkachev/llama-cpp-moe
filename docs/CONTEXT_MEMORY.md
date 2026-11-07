# Context and expert memory

Changing context capacity does not require rebuilding llama.cpp. The existing command-line context parameter and expert-cache budget are sufficient for a new launch. The selected engine already replaces cached experts during generation using the frequency policy; the model's router still chooses which experts must be evaluated.

`tools/auto_memory.py` is an optional **startup planner**, not a live memory allocator or a speedup claim. It calls the build's native `llama-fit-params` no-allocation estimator, reads the GGUF native context limit, and budgets GPU weights, context and compute buffers. It searches the expert-cache budget in 128 MiB steps, retaining 2048 MiB of estimated VRAM headroom by default. CUDA context/driver overhead and other processes can consume part of that headroom. The profile's previously tested cache size is the upper cap.

For a raw-text request it uses the native tokenizer and includes the requested output and configurable chat-template headroom. For an exact already-formatted request, pass its token count. It does not submit the prompt to the server; it plans the server capacity. Future chat growth cannot be predicted from the first request. Use a fixed target such as 262144 for a long-lived session, or explicitly restart at a larger capacity between sessions.

```bash
python3 tools/auto_memory.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen.json --ctx auto --prompt-file /path/to/prompt.txt --max-output 4096 --out /path/to/new-plan.json
python3 tools/auto_memory.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen.json --ctx 262144 --run
```

If you built only the default minimal target set, first build the two required native tools:

```bash
cmake --build "$MOE_BUILD" -j 8 --target llama-fit-params llama-tokenize
```

The first command only prints/saves a plan. `--run` starts a loopback server in a dedicated user systemd scope. The 106 GiB high / 108 GiB hard / 4 GiB swap limits and CPU affinity are specific to the measured 128 GiB host. This prototype requires one NVIDIA GPU (physical GPU 0), the native `llama-fit-params` and `llama-tokenize` tools beside the server, a single server slot, F16 KV and non-speculative profiles. It refuses to compete with an already-running llama-server. It neither restarts existing services nor connects to an application core.

No prompt truncation, lower KV precision, expert skipping or model conversion is used to make a plan fit. An impossible request fails with an explanation. It does not resize a live KV buffer, resize expert banks in flight, automatically tune RAM/NVMe residency, or persist/replay conversations through restarts. Those capabilities require a separately validated controller and backend support.

## Measured startup plans

| Model / request | Context capacity | Expert cache MiB | Estimated model+context+compute GPU MiB |
|---|---:|---:|---:|
| Qwen, 143 raw tokens + 256 output + 1024 template headroom | 4096 | 20992 | 25843 |
| Qwen, explicit 256K | 262144 | 16768 | 29336 |
| DeepSeek, explicit 256K | 262144 | 20480 | 29367 |

The free-VRAM snapshots were approximately 31440 MiB. Qwen's reported native limit was 262144; DeepSeek's was 1048576. The plans use the same selected profiles and binary without changing model arithmetic. Details are in `benchmarks/memory-plans.json`.

## Separate 128K/256K capacity verification

These are sequential single-request pilots, not an ABBA speedup experiment. The document and sampling are fixed per model; system page residency and cache warmness can differ between launches. Each prompt has exactly 4096 actual uncached input tokens and three lookup keys near its beginning, middle and end. Native EOS is enabled and a correct answer may be short. All completed rows below returned all three integer values correctly without truncating the input.

| Model | Capacity | Expert cache MiB | Input tokens/s | Input time s | Output tokens | Decode tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 262144 | 16768 | 68.14 | 60.109 | 40 | 20.56 |
| deepseek | 262144 | 20480 | 14.24 | 287.693 | 28 | 12.36 |

Full per-request timings and prompt hashes are in `benchmarks/context-capacity.json`. The approximately 77–78 tokens/s Qwen and 22–24 tokens/s DeepSeek measurements use a different, repeatedly generated short technical prompt with up to 256 output tokens; do not compare those decode numbers directly to these cold document-retrieval answers.

A configured capacity and a filled prompt are different tests. Full 128K/256K retrieval and long-input throughput remain unverified by these pilots. The newer phase-budget results in [PREFILL.md](PREFILL.md) supersede any extrapolation from these old small-ubatch pilots. Full filled 128K/256K latency remains unmeasured.

## Two phase budgets

Pass `profiles/qwen-prefill.json` or `profiles/deepseek-prefill.json` to the same planner to estimate both phases. The phase flag is removed only for native no-allocation estimates. Decode uses ubatch 16 but retains the larger SWA/recurrent state required by prefill; prefill uses the requested large ubatch. Both budgets must fit independently with the same reserve. The prefill cache is capped by its tested profile value and the selected decode cache. `phase_plan` records both sizes and estimates, and the chosen flag is applied to the launched server.

This selects budgets at startup; the optional engine patch switches those two budgets during requests. It does not grow an allocated KV context or preserve expert-cache contents across a switch. Automatically selected sizes can differ from the benchmark profiles and need their own performance measurements.
