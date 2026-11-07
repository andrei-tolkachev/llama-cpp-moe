// Native CPU regression. Each invocation reads the same synthetic GGUF and
// writes ALL logit rows, including prefill. Store greedy IDs on stdout.
#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const int tg = std::stoi(argv[3]);
    if (tg < 1 || tg > 400) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t cpu[] = {nullptr};
    mp.devices = cpu;
    mp.n_gpu_layers = 0;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 64;
    cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) return 4;
    std::ofstream rows(argv[2], std::ios::binary);
    rows.exceptions(std::ios::failbit | std::ios::badbit);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));
    int64_t pp_ns = 0, tg_ns = 0;
    for (int req = 0; req < 3; ++req) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        uint32_t pos = 0;
        const int length = req == 1 ? 13 : 41;
        std::vector<llama_token> tokens;
        for (int i = 0; i < length; ++i) tokens.push_back(3 + ((i * 17 + req % 2) % (nv - 3)));
        for (int step = 0; step <= tg; ++step) {
            llama_batch_ext_clear(batch.get());
            for (const auto token : tokens) {
                const int i = llama_batch_ext_add_token(batch.get(), 0, token);
                llama_pos pp[4] = {(llama_pos) pos, (llama_pos) pos, (llama_pos) pos, 0};
                if (i < 0 || !llama_batch_ext_set_pos(batch.get(), i, pp) ||
                    !llama_batch_ext_set_output_logits(batch.get(), i, true)) return 5;
                ++pos;
            }
            const auto start = std::chrono::steady_clock::now();
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return 6;
            llama_synchronize(ctx.get());
            const auto elapsed = std::chrono::steady_clock::now() - start;
            (step == 0 ? pp_ns : tg_ns) += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
            llama_token next = 0;
            for (size_t i = 0; i < tokens.size(); ++i) {
                const float * logits = llama_get_logits_ith(ctx.get(), int(i));
                if (!logits || !std::all_of(logits, logits + nv, [](float x) { return std::isfinite(x); })) return 7;
                rows.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
                next = int(std::max_element(logits, logits + nv) - logits);
            }
            std::cout << next << (step == tg ? '\n' : ' ');
            tokens.assign(1, next);
        }
    }
    std::cerr << "R20_SYNTHETIC_TIMINGS {\"pp_ns\":" << pp_ns << ",\"tg_ns\":" << tg_ns
              << ",\"pp_tokens\":95,\"tg_tokens\":" << 3 * tg << "}\n";
    return 0;
}
