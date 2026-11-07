# Rejected output micro-optimization

`hc-up-probe.cu` is a standalone fixture and is not compiled into the engine. Only the two absolute include paths were made portable; its kernel and test body equal the locally measured source. Results are in `benchmarks/hc-up-probe.json`. The approximately 20% improvement in this small operation is not an engine speedup: estimated token-time savings are below 0.5%. No model-throughput gain was measured.

Build against the same pinned cumulative source and CUDA libraries (set NVCC, MOE_SOURCE and MOE_BUILD to absolute paths):

```bash
"$NVCC" -O3 -DNDEBUG -std=c++17 -arch=sm_120a -use_fast_math -extended-lambda -compress-mode=size -DGGML_CUDA_USE_GRAPHS -DGGML_SHARED -I"$MOE_SOURCE/ggml/include" -I"$MOE_SOURCE/ggml/src" -I"$MOE_SOURCE/ggml/src/ggml-cuda" experiments/hc-up-probe.cu -L"$MOE_BUILD/bin" -Xlinker=-rpath -Xlinker="$MOE_BUILD/bin" -lggml-cuda -lggml-base -lggml -lcublas -lcudart -lcuda -o /tmp/moe-hc-up-probe
/tmp/moe-hc-up-probe
```

Use an architecture supported by the GPU/toolkit. The fixture requires the generic MMVQ parameter table and warp size 32 and rejects other devices. Do not benchmark it alongside a running model. It uses A/B/B/A ordering per geometry with captured repeated operations and CUDA event timing; this is not representative of all engine scheduling/launch costs.
