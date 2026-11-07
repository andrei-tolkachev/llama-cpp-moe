#pragma once

// P15C: opt-in HOST profiling of the MMQ tile selector, NOT a GPU timer.
// The observed region covers only the J / thread-count selection in mmq.cu.
// No CUDA calls, memory allocations on the hot path, or changes to GGML math.
// Each host thread emits bounded JSONL buckets on stderr at most once per
// 4096 observed calls, plus a final partial chunk when the thread exits.

#include <array>
#include <cstddef>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace llama_exp {

class mmq_select_probe final {
public:
    static constexpr std::size_t max_buckets = 512;
    static constexpr std::uint64_t report_every = 4096;

    struct sample {
        int type = -1;
        int ncols_bucket = -1;
        bool moe = false;
        std::uint64_t calls = 0;
        std::uint64_t cache_hits = 0;
        std::uint64_t select_ns = 0;
        std::uint64_t max_select_ns = 0;
    };

    static bool enabled() noexcept {
        // Exact opt-in. As with existing GGML_CUDA_MMQ_PREC, changing the env
        // inside a running process is intentionally not supported.
        static const bool value = [] {
            const char * v = std::getenv("LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return value;
    }

    static std::uint64_t clock_ns() noexcept {
        using namespace std::chrono;
        return static_cast<std::uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
    }

    static mmq_select_probe & local() noexcept {
        static thread_local mmq_select_probe instance;
        return instance;
    }

    void record(int type, bool moe, std::int64_t ncols, bool hit, std::uint64_t select_ns) noexcept {
        const int bucket = ncols <= 0 ? 0 : ncols == 1 ? 1 : ncols <= 8 ? 2 : ncols <= 64 ? 3 : ncols <= 256 ? 4 : 5;
        // Fixed open-addressing hash map: no heap, predictable bounded work,
        // no linear scan of all existing format/shape categories per call.
        sample * s = nullptr;
        const std::uint64_t h = std::uint64_t(std::uint32_t(type))*UINT64_C(0x9e3779b97f4a7c15)
                ^ std::uint64_t(bucket)*UINT64_C(0xbf58476d1ce4e5b9)
                ^ std::uint64_t(moe)*UINT64_C(0x94d049bb133111eb);
        const std::size_t first = std::size_t(h ^ (h >> 32)) & (max_buckets - 1);
        for (std::size_t attempt = 0; attempt < max_buckets; ++attempt) {
            auto & entry = samples_[(first + attempt) & (max_buckets - 1)];
            if (!entry.calls) {
                entry.type = type;
                entry.moe = moe;
                entry.ncols_bucket = bucket;
                s = &entry;
                break;
            }
            if (entry.type == type && entry.moe == moe && entry.ncols_bucket == bucket) {
                s = &entry;
                break;
            }
        }
        if (s) {
            s->calls = sat_add(s->calls, 1);
            s->cache_hits = sat_add(s->cache_hits, hit ? 1 : 0);
            s->select_ns = sat_add(s->select_ns, select_ns);
            if (s->max_select_ns < select_ns) {
                s->max_select_ns = select_ns;
            }
        } else {
            dropped_ = sat_add(dropped_, 1);
        }
        count_ += 1;
        if (count_ == report_every) {
            flush();
        }
    }

    void flush() noexcept {
        if (!count_) {
            return;
        }
        // One fprintf per JSONL record is atomically locked by the C stdio
        // implementation; no global profiler mutex on the hot path.
        for (const auto & s : samples_) {
            if (!s.calls) {
                continue;
            }
            std::fprintf(stderr,
                    "P15C_MMQ_SELECT_JSON {\"thread\":%" PRIu64 ",\"type\":%d,\"moe\":%s,"
                    "\"ncols_bucket\":%d,\"calls\":%" PRIu64 ",\"cache_hits\":%" PRIu64 ","
                    "\"select_ns\":%" PRIu64 ",\"max_select_ns\":%" PRIu64 "}\n",
                    thread_id_, s.type, s.moe ? "true" : "false", s.ncols_bucket,
                    s.calls, s.cache_hits, s.select_ns, s.max_select_ns);
        }
        if (dropped_) {
            std::fprintf(stderr,
                    "P15C_MMQ_SELECT_JSON {\"thread\":%" PRIu64 ",\"overflow\":%" PRIu64 "}\n",
                    thread_id_, dropped_);
        }
        samples_ = {};
        dropped_ = 0;
        count_ = 0;
    }

    ~mmq_select_probe() noexcept { flush(); }

    mmq_select_probe(const mmq_select_probe &) = delete;
    mmq_select_probe & operator=(const mmq_select_probe &) = delete;

private:
    mmq_select_probe() noexcept : thread_id_(next_thread_id()) {}

    static std::uint64_t next_thread_id() noexcept {
        static std::atomic<std::uint64_t> id{0};
        return id.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    static std::uint64_t sat_add(std::uint64_t left, std::uint64_t right) noexcept {
        const auto max = std::numeric_limits<std::uint64_t>::max();
        return left > max - right ? max : left + right;
    }

    std::array<sample, max_buckets> samples_{};
    std::uint64_t count_ = 0;
    std::uint64_t dropped_ = 0;
    const std::uint64_t thread_id_;
};

}  // namespace llama_exp
