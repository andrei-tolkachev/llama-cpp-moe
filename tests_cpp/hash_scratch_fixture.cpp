#include "llama-experimental/graph_stage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Host-boundary fixtures only. The generated file inserts the actual baseline
// and candidate graph-input classes and methods, not a rewritten hash algorithm.
static bool count_allocations = false;
static size_t allocations = 0;
void * operator new(size_t n) {
    if (count_allocations) { ++allocations; }
    if (void * p = std::malloc(n ? n : 1)) { return p; }
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }

#define GGML_ASSERT(x) do { if (!(x)) { throw std::runtime_error(#x); } } while (0)
using llama_token = int32_t;
static constexpr llama_token LLAMA_TOKEN_NULL = -1;
static constexpr int64_t DSV41_ENGRAM_DEAD = -1;

struct ggml_tensor {
    int id = 0;
    void * buffer = reinterpret_cast<void *>(1);
    int64_t ne[4] = {1, 1, 1, 1};
};
struct write_event {
    int id;
    size_t offset;
    std::vector<uint8_t> bytes;
    bool operator==(const write_event & b) const {
        return id == b.id && offset == b.offset && bytes == b.bytes;
    }
};
struct observation {
    std::vector<write_event> writes;
    std::vector<std::vector<int32_t>> prefetches;
    bool operator==(const observation & b) const {
        return writes == b.writes && prefetches == b.prefetches;
    }
};
static observation captured;
static bool capture = true;
static volatile uint64_t consumed_bytes = 0;
static size_t ggml_nbytes(const ggml_tensor * t) {
    return size_t(t->ne[0])*size_t(t->ne[1])*sizeof(int32_t);
}
static size_t ggml_element_size(const ggml_tensor *) { return sizeof(int32_t); }
static void ggml_backend_tensor_set(ggml_tensor * t, const void * data, size_t off, size_t n) {
    if (capture) {
        write_event event{t->id, off, std::vector<uint8_t>(n)};
        if (n) { std::memcpy(event.bytes.data(), data, n); }
        captured.writes.push_back(std::move(event));
    } else if (n) {
        consumed_bytes += static_cast<const uint8_t *>(data)[n - 1] + n;
    }
}
static void llama_prefetch_rows(ggml_tensor *, const int32_t * rows, size_t n) {
    if (capture) { captured.prefetches.emplace_back(rows, rows + n); }
    else { consumed_bytes += n; }
}

struct llama_hparams {
    uint32_t ple_ngram_size = 2;
    uint32_t ple_heads_per_ngram = 1;
    uint32_t ple_n_heads = 1;
    uint32_t ple_image_token_id = 0;
    uint32_t ple_eos_token_id = 7;
    uint32_t engram_pad_id = 0;
    int64_t dsv41_cand_block_size = 4;
    std::array<uint64_t, 8> ple_layer_multipliers{};
    std::array<uint32_t, 64> ple_head_offsets{};
    std::array<uint32_t, 64> ple_head_vocab_sizes{};
};
struct llama_ubatch {
    uint32_t n_tokens = 0;
    llama_token * token = nullptr;
    float * embd = nullptr;
    uint8_t * type = nullptr;
    int32_t * n_seq_id = nullptr;
    int64_t * pos = nullptr;
    bool mixed = false;
    bool is_mixed() const { return mixed; }
};
struct llama_kv_cache_context {
    std::vector<llama_token> previous;
    void get_prev_tokens(const llama_ubatch & batch, uint32_t n, std::vector<llama_token> & out) const {
        GGML_ASSERT(previous.size() == size_t(batch.n_tokens)*n);
        out.assign(previous.begin(), previous.end());
    }
};
struct llama_memory_hybrid_idx_context {
    llama_kv_cache_context * attn;
    const llama_kv_cache_context * get_attn() const { return attn; }
};
struct llama_kv_cache_dsv4_context {
    struct comp_plan { std::vector<int64_t> n_visible; int64_t n_kv = 0; } plan;
    llama_kv_cache_context raw;
    const llama_kv_cache_context * get_raw() const { return &raw; }
    const comp_plan & get_csa_plan(const llama_ubatch &) const { return plan; }
    const comp_plan & get_hca_plan(const llama_ubatch &) const { return plan; }
};
struct llm_graph_params { const void * mctx; llama_ubatch ubatch; };
struct llm_graph_input_i {
    virtual ~llm_graph_input_i() = default;
    virtual void set_input(const llama_ubatch *) = 0;
    virtual bool can_reuse(const llm_graph_params &) = 0;
};
struct llama_model {
    llama_hparams hparams;
    ggml_tensor * per_layer_tok_embd = nullptr;
    std::set<ggml_tensor *> can_prefetch;
};
struct llama_model_deepseek41 : llama_model {
    std::vector<int64_t> engram_token_map;
    void engram_hash(int, const llama_ubatch &, const std::vector<llama_token> &, std::vector<int32_t> &) const;
};

// @ACTUAL_ENGRAM_HASH@
namespace baseline {
// @BASELINE_QWEN@
// @BASELINE_DEEPSEEK@
}
namespace candidate {
// @CANDIDATE_QWEN@
// @CANDIDATE_DEEPSEEK@
}

static std::mt19937_64 rng(0x6b38d941);
static size_t comparisons = 0;
static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}
static void metadata(llama_hparams & h, uint32_t ng, uint32_t per, uint32_t modules) {
    h.ple_ngram_size = ng; h.ple_heads_per_ngram = per; h.ple_n_heads = (ng - 1)*per;
    require(ng*modules <= h.ple_layer_multipliers.size(), "multiplier bound");
    require(h.ple_n_heads*modules <= h.ple_head_offsets.size(), "head bound");
    for (auto & x : h.ple_layer_multipliers) { x = rng(); }
    // Include overflow in uint64 multiplication, nonzero offsets, and moduli
    // near the metadata bounds. Their primality is irrelevant to equivalence.
    for (size_t i = 0; i < h.ple_head_vocab_sizes.size(); ++i) {
        h.ple_head_vocab_sizes[i] = i % 3 == 0 ? 2147483000u : uint32_t(2 + rng()%100003);
        h.ple_head_offsets[i] = uint32_t(rng()%100);
    }
    h.ple_image_token_id = rng()%2 ? 91 : 0;
}
struct input_data {
    std::vector<llama_token> tokens, previous;
    std::vector<uint8_t> types;
    std::vector<int32_t> seqs;
    std::vector<int64_t> positions;
    float embedding = 0;
    llama_ubatch batch;
    input_data(uint32_t n, uint32_t ng, unsigned mode) : tokens(n), previous(size_t(n)*(ng-1)),
        types(n), seqs(n, 1), positions(n) {
        for (uint32_t i = 0; i < n; ++i) {
            tokens[i] = int32_t(rng()%145)-5;
            types[i] = rng()%3 == 0;
            positions[i] = i < ng ? i : int64_t(rng()%800);
        }
        for (auto & x : previous) { x = int32_t(rng()%145)-5; }
        batch = {n, mode == 1 ? nullptr : tokens.data(), mode ? &embedding : nullptr,
                 types.data(), seqs.data(), positions.data(), mode == 2};
    }
};
template<class A, class B> static void compare(A & a, B & b, const llama_ubatch & batch) {
    captured = {}; a.set_input(&batch); observation expected = std::move(captured);
    captured = {}; b.set_input(&batch); observation actual = std::move(captured);
    require(expected == actual, "host output or prefetch ordering mismatch"); ++comparisons;
}
template<class T> static size_t count_run(T & input, const llama_ubatch & batch) {
    capture = false; input.set_input(&batch); // capacity warmup is not counted
    allocations = 0; count_allocations = true;
    for (int i = 0; i < 16; ++i) { input.set_input(&batch); }
    count_allocations = false; capture = true; return allocations;
}
static void qwen_case(uint32_t ng, uint32_t per, uint32_t n, unsigned mode, bool prefetch, const char * flag) {
    if (flag) { setenv("LLAMA_EXPERIMENTAL_HASH_SCRATCH", flag, 1); }
    else { unsetenv("LLAMA_EXPERIMENTAL_HASH_SCRATCH"); }
    llama_model model; metadata(model.hparams, ng, per, 1);
    ggml_tensor table{90}, rows{1}; model.per_layer_tok_embd = &table;
    if (prefetch) { model.can_prefetch.insert(&table); }
    input_data data(n, ng, mode); llama_kv_cache_context ctx; ctx.previous = data.previous;
    rows.ne[0] = int64_t(model.hparams.ple_n_heads)*n;
    baseline::llm_graph_input_qwen4exp_ple a(model, &ctx);
    candidate::llm_graph_input_qwen4exp_ple b(model, &ctx); a.rows = b.rows = &rows;
    for (int round = 0; round < 3; ++round) {
        compare(a, b, data.batch);
        for (auto & x : ctx.previous) { x = int32_t(rng()%145)-5; }
    }
    llama_memory_hybrid_idx_context memory{&ctx}; llm_graph_params params{&memory, data.batch};
    require(a.can_reuse(params) == b.can_reuse(params), "Qwen graph reuse mismatch");
    if (flag && std::string(flag) == "1" && n == 128 && ng == 4 && per == 2 && mode == 0) {
        size_t old_count = count_run(a, data.batch), new_count = count_run(b, data.batch);
        require(new_count < old_count, "Qwen scratch did not remove allocations after warmup");
        std::cout << "Qwen allocation events / 16 calls: baseline=" << old_count << " candidate=" << new_count << '\n';
    }
}
static void deep_case(uint32_t ng, uint32_t per, uint32_t modules, uint32_t n, unsigned mode,
                      bool prefetch, const char * flag, bool omit_buffer = false) {
    if (flag) { setenv("LLAMA_EXPERIMENTAL_HASH_SCRATCH", flag, 1); }
    else { unsetenv("LLAMA_EXPERIMENTAL_HASH_SCRATCH"); }
    llama_model_deepseek41 model; metadata(model.hparams, ng, per, modules);
    model.engram_token_map.resize(128);
    for (auto & x : model.engram_token_map) { x = rng()%7 ? int64_t(rng()%200000) : -1; }
    model.engram_token_map[0] = 13;
    ggml_tensor table{90}, keep{80}, pin{81}; model.per_layer_tok_embd = &table;
    if (prefetch) { model.can_prefetch.insert(&table); }
    input_data data(n, ng, mode); llama_kv_cache_dsv4_context ctx; ctx.raw.previous = data.previous;
    ctx.plan.n_kv = 7; ctx.plan.n_visible.resize(n);
    for (auto & x : ctx.plan.n_visible) { x = int64_t(rng()%8); }
    keep.ne[2] = n; pin.ne[0] = 2; pin.ne[1] = n;
    std::vector<ggml_tensor> tensors(modules);
    baseline::llm_graph_input_dsv41 a(model, &ctx); candidate::llm_graph_input_dsv41 b(model, &ctx);
    for (uint32_t e = 0; e < modules; ++e) {
        tensors[e].id = int(e+1); tensors[e].ne[0] = int64_t(model.hparams.ple_n_heads)*n;
        if (omit_buffer && e % 2 == 0) { tensors[e].buffer = nullptr; }
        a.engram_ids.push_back(&tensors[e]); b.engram_ids.push_back(&tensors[e]);
    }
    a.engram_keep = b.engram_keep = mode ? &keep : nullptr;
    a.cand_pin = b.cand_pin = &pin; a.cand_csa = b.cand_csa = rng()%2;
    for (int round = 0; round < 3; ++round) {
        compare(a, b, data.batch);
        for (auto & x : ctx.raw.previous) { x = int32_t(rng()%145)-5; }
    }
    llm_graph_params params{&ctx, data.batch};
    require(a.can_reuse(params) == b.can_reuse(params), "DeepSeek graph reuse mismatch");
    if (flag && std::string(flag) == "1" && ng == 4 && per == 2 && modules == 2 && n == 128 && mode == 0) {
        size_t old_count = count_run(a, data.batch), new_count = count_run(b, data.batch);
        require(new_count < old_count, "DeepSeek scratch did not remove allocations after warmup");
        std::cout << "DeepSeek allocation events / 16 calls: baseline=" << old_count << " candidate=" << new_count << '\n';
    }
}
int main() {
    try {
        for (const char * flag : {static_cast<const char *>(nullptr), "0", "1", "01", "true", ""}) {
            for (unsigned mode = 0; mode < 3; ++mode) {
                for (bool prefetch : {false, true}) {
                    for (uint32_t n : {1u, 7u, 128u}) {
                        qwen_case(4, 2, n, mode, prefetch, flag);
                        deep_case(4, 2, 2, n, mode, prefetch, flag, n == 7);
                    }
                }
            }
        }
        for (uint32_t ng = 2; ng <= 8; ++ng) {
            qwen_case(ng, 1, 29, 2, true, "1");
            qwen_case(ng, 8, 29, 1, false, "1");
            deep_case(ng, 1, 1, 29, 2, true, "1");
        }
        // Exercise the large-input fallback above the 256K-value retention cap.
        qwen_case(4, 2, 44000, 2, true, "1");
        deep_case(4, 2, 2, 22000, 2, true, "1");
        std::cout << "PASS exact host output comparisons=" << comparisons << '\n';
        std::cout << "Scope: real extracted hash/input methods with host boundary fixtures; no GPU, NVMe, or model inference.\n";
        return 0;
    } catch (const std::exception & e) {
        count_allocations = false;
        std::cerr << "FAIL " << e.what() << '\n'; return 1;
    }
}
