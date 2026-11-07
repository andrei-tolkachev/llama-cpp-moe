#pragma once

// P14A: admission control for the existing P09B host CPU/GPU upload pipeline.
// This is a SOURCE-ONLY, CPU-side scheduler: it estimates the EXACT P04
// staging/transfer layout before starting a producer. It neither calls a
// backend, predicts future experts, creates CUDA streams nor changes GGUF/KV.
// All selected expert IDs have already been read back and SLRU-planned.
// A rejected candidate must use the caller's existing serial/P04 path.
//
// Opt-in: LLAMA_EXPERIMENTAL_MOE_SCHEDULER=1 (also requires the existing
// PIPELINE=1 and pinned staging flags at the call site). Default is OFF.
// The admission thresholds are deliberately conservative and must be tuned
// against real single-token decode before making any production default.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llama_exp {

enum class moe_schedule_reason : uint8_t {
    admitted,
    bank_count,
    empty_fills,
    insufficient_staging,
    too_small,
    no_coalescing,
    too_little_packing,
};

struct moe_schedule_decision {
    bool admitted = false;
    moe_schedule_reason reason = moe_schedule_reason::bank_count;
    size_t bank_count = 0;
    size_t slice_bytes = 0;
    size_t bytes = 0;             // exact H2D payload (no transfer reduction claim)
    size_t staged_bytes = 0;      // host-to-host packing estimate
    size_t upload_calls = 0;      // exact predicted P04 upload invocation count
    size_t calls_saved = 0;       // direct runs coalesced into packed transfers
    size_t packing_banks = 0;
};

class moe_upload_scheduler final {
public:
    static constexpr size_t max_banks = 8;
    static constexpr size_t alignment = 256;
    static constexpr size_t min_payload_bytes = 256 * 1024;
    static constexpr size_t min_pack_bytes = 128 * 1024;

    static bool enabled() {
        static const bool flag = [] {
            const char * s = std::getenv("LLAMA_EXPERIMENTAL_MOE_SCHEDULER");
            return s && s[0] == '1' && s[1] == '\0';
        }();
        return flag;
    }

    // WidthAt(bank) yields the exact bytes in one expert of this bank.
    // No copying and no allocation. O(n_banks * n_fills), up to 8 banks.
    // Invalid/overflowing source geometry is an exception BEFORE any backend
    // upload, never a silent fallback that could conceal a partial update.
    template <class Fill, class WidthAt>
    static moe_schedule_decision assess(
            const std::vector<Fill> & fills, size_t banks, WidthAt width_at,
            size_t staging_capacity) {
        moe_schedule_decision d;
        d.bank_count = banks;
        if (banks < 2 || banks > max_banks) return d;
        if (fills.empty()) {
            d.reason = moe_schedule_reason::empty_fills;
            return d;
        }
        // The P09 producer requires a 256-aligned slice per bank and at least
        // one full expert in each slice. Mirror its exact size calculation.
        if (staging_capacity / banks < alignment) {
            d.reason = moe_schedule_reason::insufficient_staging;
            return d;
        }
        d.slice_bytes = (staging_capacity / banks / alignment) * alignment;
        if (fills.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::length_error("MoE P14 scheduler: implausibly many fills");
        }
        for (const auto & f : fills) {
            if (f.expert < 0 || f.slot < 0) {
                throw std::invalid_argument("MoE P14 scheduler: negative expert or slot");
            }
        }

        // Count exact P04 direct runs once: whether expert AND slot advance by
        // one controls a direct H2D transfer. Only an adjacent SLOT allows us
        // to pack several otherwise discontiguous expert runs into one upload.
        const auto direct_run = [&](size_t first) {
            size_t n = 1;
            while (first + n < fills.size() &&
                    static_cast<int64_t>(fills[first+n].expert) ==
                        static_cast<int64_t>(fills[first].expert) + static_cast<int64_t>(n) &&
                    static_cast<int64_t>(fills[first+n].slot) ==
                        static_cast<int64_t>(fills[first].slot) + static_cast<int64_t>(n)) {
                ++n;
            }
            return n;
        };

        for (size_t ip = 0; ip < banks; ++ip) {
            const size_t width = width_at(ip);
            if (width == 0) {
                throw std::invalid_argument("MoE P14 scheduler: zero expert width");
            }
            if (width > d.slice_bytes) {
                d.reason = moe_schedule_reason::insufficient_staging;
                return d;
            }
            const size_t payload = mul_checked(fills.size(), width);
            d.bytes = add_checked(d.bytes, payload);
            size_t used = 0;
            bool packed_bank = false;
            for (size_t i = 0; i < fills.size();) {
                const size_t first_count = direct_run(i);
                size_t packed_count = first_count;
                size_t fragments = 1;
                const size_t fits = (d.slice_bytes - used) / width;
                // Exactly match P04's "whole run or no packing" rule.
                while (packed_count < fits && i + packed_count < fills.size() &&
                        static_cast<int64_t>(fills[i + packed_count].slot) ==
                            static_cast<int64_t>(fills[i].slot) + static_cast<int64_t>(packed_count)) {
                    const size_t count = direct_run(i + packed_count);
                    if (count > fits - packed_count) break;
                    packed_count += count;
                    ++fragments;
                }
                if (packed_count > first_count) {
                    const size_t copied = mul_checked(packed_count, width);
                    used = add_checked(used, copied);
                    d.staged_bytes = add_checked(d.staged_bytes, copied);
                    d.calls_saved = add_checked(d.calls_saved, fragments - 1);
                    i += packed_count;
                    packed_bank = true;
                } else {
                    i += first_count;
                }
                d.upload_calls = add_checked(d.upload_calls, 1);
            }
            if (packed_bank) ++d.packing_banks;
        }
        if (d.bytes < min_payload_bytes) {
            d.reason = moe_schedule_reason::too_small;
        } else if (d.calls_saved == 0) {
            d.reason = moe_schedule_reason::no_coalescing;
        } else if (d.staged_bytes < min_pack_bytes) {
            d.reason = moe_schedule_reason::too_little_packing;
        } else {
            d.reason = moe_schedule_reason::admitted;
            d.admitted = true;
        }
        return d;
    }

private:
    static size_t mul_checked(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max()/a) {
            throw std::overflow_error("MoE P14 scheduler: payload multiplication overflow");
        }
        return a*b;
    }

    static size_t add_checked(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max()-a) {
            throw std::overflow_error("MoE P14 scheduler: accumulated bytes overflow");
        }
        return a+b;
    }
};

} // namespace llama_exp
