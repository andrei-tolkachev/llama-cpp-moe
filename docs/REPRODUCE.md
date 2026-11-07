# Build, launch and reproduce

Use a separate checkout and new output directories. Do not apply patches to a running production checkout. Python 3, Git, CMake, a compatible C/C++ compiler and CUDA toolkit are required. Recorded builds used GNU 15.2, CUDA 13.4, native CPU instructions, OpenMP ON and CUDA architecture `120a` (RTX 5090). Choose the architecture supported by your own GPU/toolkit. Disabling OpenMP changes the tested configuration.

Obtain an isolated clean llama.cpp repository at commit `dde879bc4f954b809892fd0d660515f42536394a`. The accompanying `llama-upstream-pin.git.tar.gz` is an offline copy of the pinned Git repository. Inspect its paths before extracting it into an empty directory. A current llama.cpp checkout is not an interchangeable base; a forward port needs its own validation.

The pin is available in the [official upstream repository](https://github.com/ggml-org/llama.cpp/commit/dde879bc4f954b809892fd0d660515f42536394a). To fetch only that revision into a new directory:

```bash
git init /path/to/clean/pinned/llama.cpp
git -C /path/to/clean/pinned/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
git -C /path/to/clean/pinned/llama.cpp fetch --depth 1 origin dde879bc4f954b809892fd0d660515f42536394a
git -C /path/to/clean/pinned/llama.cpp checkout --detach FETCH_HEAD
```

For the offline asset (it contains a bare Git repository):

```bash
mkdir /path/to/new/upstream.git
tar -xzf llama-upstream-pin.git.tar.gz -C /path/to/new/upstream.git
git clone /path/to/new/upstream.git /path/to/clean/pinned/llama.cpp
git -C /path/to/clean/pinned/llama.cpp rev-parse HEAD
```

The printed commit must match the pin above. Verify downloaded assets against `SHA256SUMS.txt` before extraction.

From this kit directory, set absolute paths:

```bash
LLAMA_UPSTREAM=/path/to/clean/pinned/llama.cpp
MOE_SOURCE=/path/to/new/moe-source
MOE_BUILD=/path/to/new/moe-build
python3 tools/overlay.py prepare --source "$LLAMA_UPSTREAM" --destination "$MOE_SOURCE"
python3 tools/build.py --source "$MOE_SOURCE" --build "$MOE_BUILD" --arch 120a --cuda-compiler /path/to/nvcc --jobs 4
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tools/test_frequency.py --source "$MOE_SOURCE" --output /tmp/moe-frequency.cpp
c++ -std=c++17 -O2 /tmp/moe-frequency.cpp -o /tmp/moe-frequency-test
/tmp/moe-frequency-test
```

`BUILD.json` records configure/build commands and binary hashes. The default builds the server, perplexity tool and backend tests without a web UI. `--production-tools --ui-assets /path/to/existing/tools/ui/dist` preserves an existing compatible web UI and builds the production tool set. The release's actual installed checks used that full build. No command installs or restarts services automatically.

## Run a model

Point MODEL at the first GGUF shard, with all other shards alongside it. The saved short-request controls are `qwen.json` and `deepseek.json`. For automatic prefill/decode budget switching, select `qwen-prefill.json` or `deepseek-prefill.json`; see [the phase results](PREFILL.md). For example:

```bash
MODEL=/path/to/model-00001-of-00004.gguf
python3 tools/serve.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen-prefill.json --port 8081
```

The server binds loopback by default. On the measured Linux host, wrap the command in a user systemd scope with `MemoryHigh=106G`, `MemoryMax=108G`, `MemorySwapMax=4G`, and `TasksMax=256`. For example:

```bash
systemd-run --user --scope --property=MemoryHigh=106G --property=MemoryMax=108G --property=MemorySwapMax=4G --property=TasksMax=256 -- python3 tools/serve.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen-prefill.json --port 8081
```

These budgets suit that 128 GiB host only. More swap can prevent allocation failure but does not speed active expert access. Use one model on the GPU at a time and do not stop unrelated services for a benchmark.

## Cache-policy A/B/B/A

Use the **same binary, GGUF shards, prompt, context, affinity and memory limits** in every arm. All destinations must be new. This comparison changes only frequency eviction; both profiles retain private registration and backing-file cache release.

```bash
RUNS=/path/to/new/abba-results
python3 tools/screen.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen-frequency-off.json --out "$RUNS/a1" --repeats 4
python3 tools/screen.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen.json --out "$RUNS/b1" --repeats 4
python3 tools/screen.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen.json --out "$RUNS/b2" --repeats 4
python3 tools/screen.py --binary "$MOE_BUILD/bin/llama-server" --model "$MODEL" --profile profiles/qwen-frequency-off.json --out "$RUNS/a2" --repeats 4
```

Record every request, separate the first request from later requests, and compare input/output IDs as well as throughput. Native EOS is enabled. A 16-token warmup does not warm all experts. Match available memory/GPU activity, record memory pressure and clocks, and repeat on unseen prompts. Long-context, quantization and speculative decoding require separate experiments. Do not extrapolate a short prompt to filled 128K.

The portable wrapper saves raw local logs and IDs, which must be sanitized before public sharing. It clears inherited experimental/LLAMA_ARG/GGML variables, checks effective context/offload, and stops only its own process. Historical results used the original host-specific runner; the portable wrapper's real-model verification status is stated in RESULTS.md.

Historical manifests and tools remain for provenance. They do not represent additional accepted releases or a claim that every inherited flag is beneficial. Do not enable all historical flags for Qwen: that combination previously produced CUDA failures.

## CPU fusion arithmetic check

Compile the standalone regression against the same source and built GGML libraries:

```bash
c++ -std=c++17 -O2 -I"$MOE_SOURCE" -I"$MOE_SOURCE/ggml/include" tests_cpp/test-cpu-gate-up.cpp -L"$MOE_BUILD/bin" -Wl,-rpath,"$MOE_BUILD/bin" -lggml -lggml-cpu -lggml-base -o /tmp/moe-cpu-gate-up-test
LLAMA_EXPERIMENTAL_CPU_IQ3_GATHER=1 LLAMA_EXPERIMENTAL_CPU_IQ3_SIGNMASK=1 taskset -c 0-15 /tmp/moe-cpu-gate-up-test
```

The recorded run passed 1392 cases and 5,554,176 exact float comparisons, including final FFN output. Use an affinity valid on your host. See [context and memory planning](CONTEXT_MEMORY.md) for changing the context without rebuilding.

## Release archives

The release supplies `llama-cpp-moe-kit.tar.gz` (this repository), `llama-cpp-moe-source.tar.gz` (the full already-patched source without local Git metadata), and `llama-upstream-pin.git.tar.gz` (the clean offline base for replay). Verify the release-level `SHA256SUMS.txt`; the kit also has its own per-file `SHA256SUMS`.

The full source archive can be built directly after extraction with tools/build.py; do not apply the overlay to it again. For the recorded pinned Git identity and a complete patch replay, use the offline-base workflow above. Build outputs, model weights, service configuration and private deployment backups are not included.
