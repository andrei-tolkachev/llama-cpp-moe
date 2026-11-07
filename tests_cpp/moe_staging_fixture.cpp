
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#define GGML_ASSERT(x) assert(x)
#define GGML_ABORT(...) std::abort()

struct fake_backend;
struct fake_type { fake_backend * owner; bool host = true; size_t maximum = size_t(-1); };
struct fake_buffer {
    fake_backend * owner;
    fake_type * type;
    size_t size;
    bool host = true;
    bool null_base = false;
    uint8_t storage[8] = {}; // Allocation-control tests inspect metadata only.
};
struct fake_backend {
    bool async = true;
    bool host_buffer = true;
    bool missing_type = false;
    int alloc_mode = 0;
    int allocations = 0;
    int frees = 0;
    fake_type pinned {this};
    fake_type pageable {this};
};
using ggml_backend_t = fake_backend *;
using ggml_backend_dev_t = fake_backend *;
using ggml_backend_buffer_type_t = fake_type *;
using ggml_backend_buffer_t = fake_buffer *;
struct ggml_backend_dev_props { struct { bool async; bool host_buffer; } caps; };
static void ggml_backend_buffer_free(fake_buffer * p) {
    if (p) { p->owner->frees++; delete p; }
}
struct fake_deleter { void operator()(fake_buffer * p) const { ggml_backend_buffer_free(p); } };
using ggml_backend_buffer_ptr = std::unique_ptr<fake_buffer, fake_deleter>;
static ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t p) { return p; }
static void ggml_backend_dev_get_props(ggml_backend_dev_t p, ggml_backend_dev_props * props) {
    props->caps.async = p->async; props->caps.host_buffer = p->host_buffer;
}
static fake_type * ggml_backend_dev_host_buffer_type(fake_backend * p) { return p->missing_type ? nullptr : &p->pinned; }
static bool ggml_backend_buft_is_host(fake_type * p) { return p->host; }
static size_t ggml_backend_buft_get_max_size(fake_type * p) { return p->maximum; }
static fake_buffer * ggml_backend_buft_alloc_buffer(fake_type * t, size_t size) {
    auto * p = t->owner;
    p->allocations++;
    if (p->alloc_mode == 1) { return nullptr; }
    auto * result = new fake_buffer {p, t, size};
    if (p->alloc_mode == 2) { result->type = &p->pageable; }
    if (p->alloc_mode == 3) { result->size = size - 1; }
    if (p->alloc_mode == 4) { result->size = size_t(64)*1024*1024 + 1; }
    if (p->alloc_mode == 5) { result->null_base = true; }
    if (p->alloc_mode == 6) { result->host = false; }
    return result;
}
static fake_type * ggml_backend_buffer_get_type(fake_buffer * p) { return p->type; }
static bool ggml_backend_buffer_is_host(fake_buffer * p) { return p->host; }
static size_t ggml_backend_buffer_get_size(fake_buffer * p) { return p->size; }
static void * ggml_backend_buffer_get_base(fake_buffer * p) { return p->null_base ? nullptr : p->storage; }

// @ACTUAL_STAGING_HELPERS@

static void check_env() {
    bool invalid = true;
    assert(llama_moe_cache_staging_limit(nullptr, invalid) == 0 && !invalid);
    for (const char * text : {"", "0", "00"}) {
        assert(llama_moe_cache_staging_limit(text, invalid) == 0 && !invalid);
    }
    for (const char * text : {"1x", "-1", "+1", " 1", "1 ", "65", "999999999999999999999999999999999999", "1.0"}) {
        assert(llama_moe_cache_staging_limit(text, invalid) == 0 && invalid);
    }
    assert(llama_moe_cache_staging_limit("1", invalid) == 1024*1024 && !invalid);
    assert(llama_moe_cache_staging_limit("032", invalid) == size_t(32)*1024*1024 && !invalid);
    assert(llama_moe_cache_staging_limit("64", invalid) == size_t(64)*1024*1024 && !invalid);
}

static void check_allocation() {
    constexpr size_t limit = 1024*1024;
    {
        fake_backend b;
        assert(!llama_moe_cache_alloc_staging(&b, limit, true));
        assert(!llama_moe_cache_alloc_staging(&b, 0, false));
        assert(b.allocations == 0);
    }
    for (int eligibility = 0; eligibility < 5; ++eligibility) {
        fake_backend b;
        if (eligibility == 0) b.async = false;
        if (eligibility == 1) b.host_buffer = false;
        if (eligibility == 2) b.missing_type = true;
        if (eligibility == 3) b.pinned.host = false;
        if (eligibility == 4) b.pinned.maximum = limit - 1;
        assert(!llama_moe_cache_alloc_staging(&b, limit, false));
        assert(b.allocations == 0);
    }
    for (int mode = 1; mode <= 6; ++mode) {
        fake_backend b;
        b.alloc_mode = mode;
        assert(!llama_moe_cache_alloc_staging(&b, limit, false));
        assert(b.allocations == 1);
        assert(b.frees == (mode == 1 ? 0 : 1));
    }
    fake_backend a, b;
    auto first = llama_moe_cache_alloc_staging(&a, limit, false);
    auto second = llama_moe_cache_alloc_staging(&b, limit, false);
    assert(first && second && first.get() != second.get());
    assert(ggml_backend_buffer_get_base(first.get()) != ggml_backend_buffer_get_base(second.get()));
    first.reset(); second.reset();
    assert(a.frees == 1 && b.frees == 1);
}

struct pending_upload { const uint8_t * data; size_t bank, offset, size; };
struct case_result { size_t calls = 0, saved = 0, staged = 0; };

static size_t direct_calls(const std::vector<moe_cache_slru::fill> & fills) {
    size_t calls = 0;
    for (size_t i = 0; i < fills.size(); ++i) {
        calls += i == 0 || fills[i].expert != fills[i-1].expert + 1 || fills[i].slot != fills[i-1].slot + 1;
    }
    return calls;
}

static case_result check_bytes(const std::vector<moe_cache_slru::fill> & fills,
        const std::vector<size_t> & strides, size_t capacity, size_t initial_used = 0, bool null_staging = false) {
    constexpr size_t n_experts = 64, n_slots = 64, guard = 16;
    assert(initial_used <= capacity);
    std::vector<uint8_t> arena(capacity + 2*guard, 0xda);
    uint8_t * staging = arena.data() + guard;
    std::vector<std::vector<uint8_t>> sources(strides.size()), banks(strides.size()), expected(strides.size());
    std::vector<pending_upload> pending;
    size_t used = initial_used;
    case_result total;
    for (size_t bank = 0; bank < strides.size(); ++bank) {
        const size_t stride = strides[bank];
        sources[bank].resize(n_experts*stride);
        for (size_t i = 0; i < sources[bank].size(); ++i) {
            sources[bank][i] = uint8_t((i*17 + i/stride*37 + bank*71) % 251);
        }
        banks[bank].assign(n_slots*stride, 0xcc);
        expected[bank] = banks[bank];
        for (const auto & fill : fills) {
            std::memcpy(expected[bank].data() + size_t(fill.slot)*stride,
                        sources[bank].data() + size_t(fill.expert)*stride, stride);
        }
        const size_t before_used = used, before_calls = pending.size();
        const auto uploaded = llama_moe_cache_upload_experts(sources[bank].data(), stride, fills,
            null_staging ? nullptr : staging, capacity, used,
            [&](const uint8_t * p, size_t offset, size_t bytes) {
                assert(offset + bytes <= banks[bank].size());
                pending.push_back({p, bank, offset, bytes});
            });
        assert(used <= capacity && used >= before_used);
        assert(uploaded.bytes == fills.size()*stride);
        assert(uploaded.staged_bytes == used - before_used);
        assert(uploaded.calls == pending.size() - before_calls);
        assert(uploaded.calls + uploaded.calls_saved == direct_calls(fills));
        total.calls += uploaded.calls;
        total.saved += uploaded.calls_saved;
        total.staged += uploaded.staged_bytes;
    }
    // All banks have queued their transfers before ANY source pointer is read.
    // An incorrect per-bank cursor reset will overwrite an earlier bank here.
    size_t last_staged_end = initial_used;
    const auto begin = reinterpret_cast<uintptr_t>(staging);
    for (const auto & upload : pending) {
        const auto address = reinterpret_cast<uintptr_t>(upload.data);
        if (address >= begin && address < begin + capacity) {
            const size_t offset = size_t(address - begin);
            assert(offset >= last_staged_end && offset + upload.size <= used);
            last_staged_end = offset + upload.size;
        }
        std::memcpy(banks[upload.bank].data() + upload.offset, upload.data, upload.size);
    }
    assert(banks == expected);
    assert(std::all_of(arena.begin(), arena.begin() + guard + initial_used, [](uint8_t x) { return x == 0xda; }));
    assert(std::all_of(arena.begin() + guard + used, arena.end(), [](uint8_t x) { return x == 0xda; }));
    return total;
}

static void check_deferred_backends() {
    const std::vector<moe_cache_slru::fill> fills {{1, 0}, {4, 1}, {8, 2}};
    constexpr size_t stride = 17;
    std::array<std::vector<uint8_t>, 2> source, staging, bank;
    std::array<size_t, 2> used {{0, 0}};
    std::array<std::vector<pending_upload>, 2> pending;
    for (size_t dev = 0; dev < 2; ++dev) {
        source[dev].resize(10*stride);
        std::iota(source[dev].begin(), source[dev].end(), uint8_t(dev*71));
        staging[dev].resize(3*stride);
        bank[dev].resize(3*stride);
        const auto result = llama_moe_cache_upload_experts(source[dev].data(), stride, fills,
            staging[dev].data(), staging[dev].size(), used[dev],
            [&](const uint8_t * p, size_t offset, size_t bytes) { pending[dev].push_back({p, dev, offset, bytes}); });
        assert(result.calls == 1 && result.calls_saved == 2);
    }
    auto drain = [&](size_t dev) {
        for (const auto & p : pending[dev]) {
            std::memcpy(bank[dev].data() + p.offset, p.data, p.size);
        }
        pending[dev].clear();
        for (const auto & f : fills) {
            assert(std::memcmp(bank[dev].data() + size_t(f.slot)*stride,
                               source[dev].data() + size_t(f.expert)*stride, stride) == 0);
        }
    };
    drain(0); // This stands for the existing synchronization of backend zero.
    used[0] = 0;
    std::fill(source[0].begin(), source[0].end(), 0x5b);
    llama_moe_cache_upload_experts(source[0].data(), stride, fills,
        staging[0].data(), staging[0].size(), used[0],
        [&](const uint8_t * p, size_t offset, size_t bytes) { pending[0].push_back({p, 0, offset, bytes}); });
    drain(1); // Backend one's previous pointers were not overwritten by reuse of zero.
    drain(0);
}

int main() {
    check_env();
    check_allocation();
    const std::vector<moe_cache_slru::fill> gaps {{1, 0}, {4, 1}, {8, 2}};
    auto result = check_bytes(gaps, {144, 210, 272}, 3*(144 + 210 + 272));
    assert(result.calls == 3 && result.saved == 6 && result.staged == 3*(144 + 210 + 272));
    result = check_bytes(gaps, {17, 31, 97}, 3*17); // Exhausted after bank zero; no reuse before completion.
    assert(result.calls == 7 && result.saved == 2 && result.staged == 3*17);
    result = check_bytes(gaps, {17, 31}, 2*17); // A partial run may be packed; the remainder stays direct.
    assert(result.calls == 5 && result.saved == 1);
    result = check_bytes(gaps, {17}, 16); // Cannot fit even one expert.
    assert(result.calls == 3 && result.saved == 0 && result.staged == 0);
    result = check_bytes(gaps, {17}, 1024, 0, true);
    assert(result.calls == 3 && result.saved == 0 && result.staged == 0);
    result = check_bytes({{1, 0}, {2, 1}, {3, 2}}, {17, 144}, 4096);
    assert(result.calls == 2 && result.saved == 0 && result.staged == 0); // Already one direct run per bank.
    result = check_bytes({{1, 3}, {4, 0}, {8, 7}}, {17}, 4096);
    assert(result.calls == 3 && result.saved == 0); // Nonadjacent destination slots are never bridged.
    result = check_bytes({}, {17, 210}, 0);
    assert(result.calls == 0 && result.staged == 0);
    // The budget must not cut through the second original direct run. Doing
    // so would save no call yet copy bytes, and overcount calls_saved by one.
    result = check_bytes({{1, 0}, {4, 1}, {5, 2}, {6, 3}}, {17}, 3*17);
    assert(result.calls == 2 && result.saved == 0 && result.staged == 0);
    result = check_bytes({{1, 0}, {4, 1}, {5, 2}, {6, 3}, {8, 4}}, {17}, 4*17);
    assert(result.calls == 2 && result.saved == 1 && result.staged == 4*17);
    check_deferred_backends();

    std::mt19937 rng(0x50404);
    const std::array<size_t, 8> strides {{1, 3, 17, 144, 210, 256, 272, 1024}};
    size_t randomized_saved_calls = 0;
    for (size_t trial = 0; trial < 4000; ++trial) {
        std::vector<int32_t> experts(48), slots(48);
        std::iota(experts.begin(), experts.end(), 0);
        std::iota(slots.begin(), slots.end(), 0);
        std::shuffle(experts.begin(), experts.end(), rng);
        if (trial % 2) std::shuffle(slots.begin(), slots.end(), rng);
        const size_t count = rng() % 40;
        experts.resize(count);
        std::sort(experts.begin(), experts.end());
        std::vector<moe_cache_slru::fill> fills;
        for (size_t i = 0; i < count; ++i) fills.push_back({experts[i], slots[i]});
        std::vector<size_t> sizes {strides[rng() % strides.size()], strides[rng() % strides.size()], 13 + rng() % 101};
        const size_t required = count*std::accumulate(sizes.begin(), sizes.end(), size_t(0));
        const size_t capacity = rng() % (required + 31);
        const size_t initial = capacity == 0 ? 0 : rng() % (capacity + 1);
        randomized_saved_calls += check_bytes(fills, sizes, capacity, initial).saved;
    }
    assert(randomized_saved_calls > 0);
    std::printf("{\"status\":\"PASS\",\"randomized_byte_cases\":4000,\"deterministic_byte_cases\":10,"
                "\"deferred_backend_lifetime_cases\":1,\"allocation_control_cases\":14,\"env_cases\":15,"
                "\"randomized_upload_calls_saved\":%zu,\"gpu_or_inference_test\":false}\n", randomized_saved_calls);
}
