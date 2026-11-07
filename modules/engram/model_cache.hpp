#pragma once

// P13: model-owned, exact-raw-row Engram cache. The source is a *borrowed*
// immutable GGUF host mapping. The owning llama_model_deepseek41 must call
// close() in its derived destructor, before the base releases tensor buffers.
// Graph-stage instances may share this cache but own their staging scratch.
// No second disk format, speculative hash, lossy conversion or global worker.

#include "mapped_rows.hpp"
#include "row_cache.hpp"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llama_exp {

class engram_model_cache final {
public:
    static constexpr size_t default_data_budget = size_t(16) << 20;
    static constexpr size_t async_queue_limit = 128;

    engram_model_cache(mapped_rows view, uint64_t total_rows,
                       size_t budget = default_data_budget,
                       size_t pending_limit = 0)
        : view_(view), total_rows_(total_rows), trace_(trace_enabled()),
          cache_([this](uint64_t id, uint8_t * dest, size_t bytes) {
              return view_(id, dest, bytes);
          }, total_rows, view.payload_bytes, budget, pending_limit) {
        if (!total_rows_ || total_rows_ - 1 >
                (view_.mapped_bytes - view_.payload_bytes) / view_.stride) {
            throw std::invalid_argument("Engram source extent is smaller than tensor rows");
        }
    }

    engram_model_cache(const engram_model_cache &) = delete;
    engram_model_cache & operator=(const engram_model_cache &) = delete;
    ~engram_model_cache() {
        close();
        if (trace_) {
            const auto stats = cache_.snapshot();
            // One record per cache object after draining. Reads count mapping
            // accesses; Linux page-cache hits are possible on every read.
            std::fprintf(stderr, "R20_ENGRAM_JSON {\"table_rows\":%" PRIu64
                    ",\"row_bytes\":%zu,\"stage_calls\":%" PRIu64 ",\"rows_completed\":%" PRIu64
                    ",\"stage_host_ns_sum\":%" PRIu64 ",\"cache_hits\":%" PRIu64
                    ",\"foreground_reads\":%" PRIu64 ",\"worker_reads\":%" PRIu64
                    ",\"already_filled\":%" PRIu64 ",\"queue_dropped\":%" PRIu64
                    ",\"wait_hits\":%" PRIu64 ",\"read_failures\":%" PRIu64 "}\n",
                    total_rows_, view_.payload_bytes, calls_.load(), rows_done_.load(),
                    host_ns_.load(), stats.hit, stats.synchronous_reads, stats.background_reads,
                    stats.already_filled, stats.queue_dropped, stats.wait_hits, stats.read_failures);
        }
    }

    bool matches(mapped_rows view, uint64_t rows) const noexcept {
        return view_.base == view.base && view_.mapped_bytes == view.mapped_bytes &&
               view_.stride == view.stride && view_.payload_bytes == view.payload_bytes &&
               total_rows_ == rows;
    }

    // Enqueue only validated row IDs, copying every ID into a bounded queue.
    // A queue-full or disabled async worker is never an inference failure:
    // foreground stage() still reads the exact source bytes.
    size_t prefetch(const int32_t * ids, size_t count) {
        return cache_.prefetch_i32(ids, count);
    }

    engram_row_cache::read_result get(uint64_t id, uint8_t * dest, size_t n) {
        return cache_.get(id, dest, n, std::chrono::milliseconds(1));
    }

    bool get_many(const int32_t * ids, size_t count, uint8_t * dest, size_t bytes) {
        if (!trace_) return cache_.get_many_i32(ids, count, dest, bytes, std::chrono::milliseconds(1));
        const auto start = std::chrono::steady_clock::now();
        const bool ok = cache_.get_many_i32(ids, count, dest, bytes, std::chrono::milliseconds(1));
        const auto elapsed = std::chrono::steady_clock::now() - start;
        calls_.fetch_add(1, std::memory_order_relaxed);
        if (ok) rows_done_.fetch_add(count, std::memory_order_relaxed);
        host_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
                           std::memory_order_relaxed);
        return ok;
    }

    size_t row_bytes() const noexcept { return view_.payload_bytes; }
    uint64_t total_rows() const noexcept { return total_rows_; }
    engram_row_cache::counters counters() const { return cache_.snapshot(); }

    // Idempotent. Must complete BEFORE tensor buffers and mmap are released.
    // A blocked page fault cannot be force-cancelled: close() waits for it.
    void close() noexcept { cache_.close(); }

private:
    static bool trace_enabled() noexcept {
        const char * value = std::getenv("LLAMA_EXPERIMENTAL_ENGRAM_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }
    const mapped_rows view_;
    const uint64_t total_rows_;
    const bool trace_;
    std::atomic<uint64_t> calls_{0}, rows_done_{0}, host_ns_{0};
    engram_row_cache cache_;
};

} // namespace llama_exp
