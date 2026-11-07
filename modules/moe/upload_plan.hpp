#pragma once

// P12: one immutable layout of exact MoE fill runs per cache update.
// The same plan can be read concurrently by P09B producer and caller threads,
// and reused by every expert tensor bank with its own expert_size/staging budget.
// No speculative routing, truncation, tensor/quantization changes or backend calls.
// The caller owns source bytes, staging, and the plan for the entire execute().
// C++17, no llama.cpp / GGML dependencies. Opt-in and disabled by default.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llama_exp {

struct moe_fill_plan_stats {
    size_t bytes = 0;
    size_t staged_bytes = 0;
    size_t calls = 0;
    size_t calls_saved = 0;
};

class moe_fill_plan final {
public:
    struct run {
        size_t expert;
        size_t slot;
        size_t count;
    };

    static bool enabled() {
        static const bool value = [] {
            const char * env = std::getenv("LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN");
            return env && env[0] == '1' && env[1] == '\0';
        }();
        return value;
    }

    template <class Fill>
    explicit moe_fill_plan(const std::vector<Fill> & fills) {
        runs_.reserve(fills.size());
        for (size_t i = 0; i < fills.size();) {
            // The SLRU fill planner emits nonnegative 32-bit IDs and slots.
            // Reject malformed input before any copies are submitted.
            if (fills[i].expert < 0 || fills[i].slot < 0) {
                throw std::invalid_argument("MoE shared fill plan: negative expert or slot");
            }
            const size_t expert = static_cast<size_t>(fills[i].expert);
            const size_t slot   = static_cast<size_t>(fills[i].slot);
            size_t n = 1;
            while (i + n < fills.size() &&
                   static_cast<int64_t>(fills[i + n].expert) == static_cast<int64_t>(fills[i].expert) + static_cast<int64_t>(n) &&
                   static_cast<int64_t>(fills[i + n].slot)   == static_cast<int64_t>(fills[i].slot) + static_cast<int64_t>(n)) {
                ++n;
            }
            runs_.push_back({ expert, slot, n });
            i += n;
        }
    }

    size_t run_count() const noexcept { return runs_.size(); }

    // Exact P04 semantics: start with the full direct run (even if it cannot
    // fit staging), then append whole next runs only when slots are adjacent
    // and the entire run fits. Never split a direct run to fill a slice.
    // Every bank has its own width and own cursor: the same immutable plan is
    // safe for concurrent prepares with disjoint output slices.
    template <class Upload>
    moe_fill_plan_stats upload(const uint8_t * source, size_t expert_size,
                               uint8_t * staging, size_t capacity, size_t & used,
                               Upload upload) const {
        if (!source || expert_size == 0 || used > capacity) {
            throw std::invalid_argument("MoE shared fill plan: invalid upload parameters");
        }
        moe_fill_plan_stats stats;
        for (size_t i = 0; i < runs_.size();) {
            const auto & start = runs_[i];
            const size_t fits = staging ? (capacity - used)/expert_size : 0;
            size_t packed_n = start.count;
            size_t run_end = i + 1;
            while (packed_n < fits && run_end < runs_.size()) {
                const auto & next = runs_[run_end];
                if (start.slot > std::numeric_limits<size_t>::max() - packed_n ||
                    next.slot != start.slot + packed_n ||
                    next.count > fits - packed_n) {
                    break;
                }
                packed_n += next.count;
                ++run_end;
            }
            if (run_end > i + 1) {
                // All multiplications are bounded by fits; the stage cursor
                // cannot wrap or overrun its slice.
                for (size_t k = i, offset = 0; k < run_end; ++k) {
                    const auto & r = runs_[k];
                    copy_to_stage(staging + used + offset*expert_size,
                                  source, r.expert, r.count, expert_size);
                    offset += r.count;
                }
                const size_t bytes = packed_n*expert_size;
                upload(staging + used, checked_offset(start.slot, expert_size), bytes);
                used += bytes;
                stats.bytes        += bytes;
                stats.staged_bytes += bytes;
                ++stats.calls;
                stats.calls_saved += run_end - i - 1;
                i = run_end;
            } else {
                const size_t bytes = checked_offset(start.count, expert_size);
                upload(checked_source(source, start.expert, expert_size),
                       checked_offset(start.slot, expert_size), bytes);
                stats.bytes += bytes;
                ++stats.calls;
                ++i;
            }
        }
        return stats;
    }

private:
    static size_t checked_offset(size_t value, size_t width) {
        if (value > std::numeric_limits<size_t>::max()/width) {
            throw std::overflow_error("MoE shared fill plan: offset overflow");
        }
        return value*width;
    }

    static const uint8_t * checked_source(const uint8_t * source, size_t expert, size_t width) {
        return source + checked_offset(expert, width);
    }

    static void copy_to_stage(uint8_t * dst, const uint8_t * source,
                              size_t expert, size_t count, size_t width) {
        std::memcpy(dst, checked_source(source, expert, width), checked_offset(count, width));
    }

    std::vector<run> runs_;
};

} // namespace llama_exp
