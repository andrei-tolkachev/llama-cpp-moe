#pragma once

// P15B: pure HOST-SIDE memoization of GGML-CUDA MMQ MoE tile selection.
// The key contains every argument used by the original J/threads solver.
// Collisions cause recomputation, never an approximate result.
// No CUDA device state or GGUF weights are stored in this cache.

#include <cstddef>
#include <cstdint>

namespace llama_exp {

struct mmq_tile_key {
    int32_t type = 0;
    int64_t ncols = 0;
    bool fallback = false;
    int32_t cc = 0;
    uint64_t shared_bytes = 0;
    int32_t precision = 0;

    bool operator==(const mmq_tile_key & rhs) const noexcept {
        return type == rhs.type && ncols == rhs.ncols && fallback == rhs.fallback &&
               cc == rhs.cc && shared_bytes == rhs.shared_bytes && precision == rhs.precision;
    }
};

struct mmq_tile_result {
    int J = 0;
    int nthreads = 0;
};

template <size_t Slots = 32>
class mmq_tile_cache {
    static_assert(Slots > 0 && (Slots & (Slots - 1)) == 0,
                  "MMQ cache Slots must be a nonzero power of two");

    struct entry {
        mmq_tile_key key{};
        mmq_tile_result result{};
        bool valid = false;
    };
    entry entries_[Slots] = {};

    static uint64_t mix(uint64_t x) noexcept {
        // Fixed-width unsigned operations, independent of signed overflow.
        x ^= x >> 30;
        x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x *= UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }
    static size_t index(const mmq_tile_key & k) noexcept {
        uint64_t h = UINT64_C(0x9e3779b97f4a7c15);
        h = mix(h ^ uint64_t(uint32_t(k.type)));
        h = mix(h ^ uint64_t(k.ncols));
        h = mix(h ^ uint64_t(k.fallback));
        h = mix(h ^ uint64_t(uint32_t(k.cc)));
        h = mix(h ^ k.shared_bytes);
        h = mix(h ^ uint64_t(uint32_t(k.precision)));
        return size_t(h) & (Slots - 1);
    }

public:
    template <typename Compute>
    mmq_tile_result get_or_compute(const mmq_tile_key & key, Compute && compute) {
        entry & e = entries_[index(key)];
        if (e.valid && e.key == key) {
            return e.result;
        }
        // Never install a partially calculated entry if compute throws.
        const mmq_tile_result result = compute();
        e.key = key;
        e.result = result;
        e.valid = true;
        return result;
    }
};

} // namespace llama_exp
