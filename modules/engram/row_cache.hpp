#pragma once

// Source-only experimental building block for exact GGUF Engram/PLE row bytes.
// Used by the optional DeepSeek 4.1 graph stage. The caller must close()
// the cache before unmapping it. No GGUF conversion, decoded-row cache, or
// extra tensor representation is involved. C++17, no dependencies on ggml.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llama_exp {

class engram_row_cache final {
public:
    using row_id = uint64_t;
    // Reader is called with *exactly* row_bytes() of writable storage.
    // Reader can return false or throw on an I/O fault; no partial row is cached.
    // Calls from the worker and foreground reader are serialized.
    using reader_type = std::function<bool(row_id, uint8_t *, size_t)>;

    enum class read_result { cache_hit, source_read, io_error, closed, invalid_row, invalid_output };

    struct counters {
        uint64_t hit = 0;
        uint64_t synchronous_reads = 0;
        uint64_t background_reads = 0;
        uint64_t read_failures = 0;
        uint64_t queued = 0;
        uint64_t duplicate_prefetch = 0;
        uint64_t already_filled = 0;
        uint64_t queue_dropped = 0;
        uint64_t evicted = 0;
        uint64_t wait_hits = 0;
        size_t resident_rows = 0;
        size_t queued_or_inflight = 0;
        size_t resident_data_bytes = 0;
        size_t row_capacity = 0;
        bool stopped = false;
    };

    // max_data_bytes is a hard limit for retained payload, not an accounting of
    // std::unordered_map/list/allocator metadata. queue_capacity bounds pending
    // row IDs; 0 disables the worker and the queue but permits sync caching.
    explicit engram_row_cache(reader_type reader, row_id total_rows,
                              size_t row_bytes, size_t max_data_bytes,
                              size_t queue_capacity)
            : reader_(std::move(reader)), total_rows_(total_rows), row_bytes_(row_bytes),
              row_capacity_(row_bytes ? max_data_bytes / row_bytes : 0),
              queue_capacity_(queue_capacity) {
        if (!reader_ || total_rows_ == 0 || row_bytes_ == 0 || row_capacity_ == 0 ||
                row_bytes_ > (size_t(1) << 20) || max_data_bytes > (size_t(64) << 20) ||
                queue_capacity_ > (size_t(1) << 16)) {
            throw std::invalid_argument("invalid Engram row cache geometry/budget");
        }
        if (queue_capacity_) {
            worker_ = std::thread(&engram_row_cache::run, this);
        }
    }

    engram_row_cache(const engram_row_cache &) = delete;
    engram_row_cache & operator=(const engram_row_cache &) = delete;
    engram_row_cache(engram_row_cache &&) = delete;
    engram_row_cache & operator=(engram_row_cache &&) = delete;

    ~engram_row_cache() { close(); }

    size_t row_bytes() const noexcept { return row_bytes_; }
    size_t row_capacity() const noexcept { return row_capacity_; }

    // Nonblocking. The queue never holds pointers supplied by the caller.
    // Full queue requests are dropped, NOT delayed, preserving decode latency.
    size_t prefetch(const row_id * ids, size_t n) {
        if (!ids || n == 0 || queue_capacity_ == 0) return 0;
        std::unique_lock<std::mutex> lock(mu_);
        if (stopping_) return 0;
        size_t submitted = 0;
        for (size_t i = 0; i < n; ++i) {
            const row_id id = ids[i];
            if (id >= total_rows_) continue;
            if (entries_.count(id) || pending_.count(id)) {
                ++stats_.duplicate_prefetch;
                continue;
            }
            if (pending_.size() >= queue_capacity_) {
                ++stats_.queue_dropped;
                continue;
            }
            pending_.insert(id);
            try {
                queue_.push_back(id);
            } catch (...) {
                pending_.erase(id);
                throw;
            }
            ++submitted;
        }
        stats_.queued += submitted;
        if (submitted) cv_work_.notify_one();
        return submitted;
    }

    // P13: no temporary std::vector for per-graph int32_t hashed row IDs.
    // No source pointer survives this call. Duplicates/negative IDs are ignored.
    size_t prefetch_i32(const int32_t * ids, size_t n) {
        if (!ids || n == 0 || queue_capacity_ == 0) return 0;
        std::unique_lock<std::mutex> lock(mu_);
        if (stopping_) return 0;
        size_t submitted = 0;
        for (size_t i = 0; i < n; ++i) {
            if (ids[i] < 0) continue;
            const row_id id = uint64_t(ids[i]);
            if (id >= total_rows_) continue;
            if (entries_.count(id) || pending_.count(id)) {
                ++stats_.duplicate_prefetch;
                continue;
            }
            if (pending_.size() >= queue_capacity_) {
                ++stats_.queue_dropped;
                continue;
            }
            pending_.insert(id);
            try {
                queue_.push_back(id);
            } catch (...) {
                pending_.erase(id);
                throw;
            }
            ++submitted;
        }
        stats_.queued += submitted;
        if (submitted) cv_work_.notify_one();
        return submitted;
    }

    // Foreground read provides an exact fallback independent of prefetch state.
    // A zero wait never blocks on a pending prefetch condition (but may need to
    // wait for the shared source reader mutex). A bounded wait may use a newly
    // completed prefetch, otherwise it falls through to a source read.
    read_result get(row_id id, uint8_t * dest, size_t dest_size,
                    std::chrono::milliseconds max_wait = std::chrono::milliseconds(0)) {
        if (!dest || dest_size < row_bytes_) return read_result::invalid_output;
        if (id >= total_rows_) return read_result::invalid_row;
        std::unique_lock<std::mutex> lock(mu_);
        return get_locked(lock, id, dest, max_wait);
    }

    // Called for already hashed GGML rows in logical gather order. Keep the
    // metadata lock over a short run of hits. get_locked releases it before
    // all waits and source reads; the chunk bound also lets close/other graphs
    // make progress on all-hit prefill batches.
    bool get_many_i32(const int32_t * ids, size_t n, uint8_t * dest, size_t bytes,
                      std::chrono::milliseconds max_wait) {
        if (!ids || !dest || n == 0 || n > bytes / row_bytes_) return false;
        const size_t chunk = std::min<size_t>(32, std::max<size_t>(1, 65536 / row_bytes_));
        for (size_t first = 0; first < n; ) {
            const size_t count = std::min(chunk, n - first);
            std::unique_lock<std::mutex> lock(mu_);
            for (size_t i = first; i < first + count; ++i) {
                if (ids[i] < 0 || row_id(ids[i]) >= total_rows_) return false;
                const auto result = get_locked(lock, row_id(ids[i]), dest + i * row_bytes_, max_wait);
                if (result != read_result::cache_hit && result != read_result::source_read) return false;
            }
            first += count;
        }
        return true;
    }

    // Close is idempotent and waits for worker plus in-flight get() readers.
    // This must precede unmapping/releasing the source tensor. Blocking source
    // readers cannot be force-cancelled; the owner must bound their I/O time.
    void close() noexcept {
        std::lock_guard<std::mutex> shutdown(shutdown_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!stopping_) {
                stopping_ = true;
                queue_.clear();
                cv_work_.notify_all();
                cv_ready_.notify_all();
            }
        }
        if (worker_.joinable()) worker_.join();
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_ready_.wait(lock, [&] { return active_reads_ == 0; });
            entries_.clear();
            lru_.clear();
            pending_.clear();
        }
    }

    counters snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        counters result = stats_;
        result.resident_rows = entries_.size();
        result.queued_or_inflight = pending_.size();
        result.resident_data_bytes = entries_.size() * row_bytes_;
        result.row_capacity = row_capacity_;
        result.stopped = stopping_;
        return result;
    }

private:
    read_result get_locked(std::unique_lock<std::mutex> &lock, row_id id, uint8_t *dest,
                           std::chrono::milliseconds max_wait) {
        if (stopping_) return read_result::closed;
        if (fetch_hit(id, dest)) return read_result::cache_hit;

        // A waiter releases mu_ in wait_until; it is already an in-flight
        // get(). Keep it registered until it has released reader_mu_ as well.
        ++active_reads_;
        struct active_read {
            engram_row_cache & cache;
            std::unique_lock<std::mutex> & lock;
            ~active_read() {
                if (!lock.owns_lock()) lock.lock();
                if (--cache.active_reads_ == 0 && cache.stopping_) cache.cv_ready_.notify_all();
            }
        } registered{*this, lock};
        if (max_wait.count() > 0 && pending_.count(id)) {
            const auto end = std::chrono::steady_clock::now() + max_wait;
            cv_ready_.wait_until(lock, end, [&] {
                return stopping_ || !pending_.count(id) || entries_.count(id);
            });
            if (stopping_) return read_result::closed;
            if (fetch_hit(id, dest)) {
                ++stats_.wait_hits;
                return read_result::cache_hit;
            }
        }

        lock.unlock();
        std::lock_guard<std::mutex> source_lock(reader_mu_);
        lock.lock();
        if (stopping_) return read_result::closed;
        if (fetch_hit(id, dest)) return read_result::cache_hit;

        lock.unlock();
        bool ok = false;
        try { ok = reader_(id, dest, row_bytes_); } catch (...) { ok = false; }
        lock.lock();
        if (stopping_) return read_result::closed;
        if (!ok) {
            ++stats_.read_failures;
            return read_result::io_error;
        }
        ++stats_.synchronous_reads;
        insert(id, dest);
        cv_ready_.notify_all();
        return read_result::source_read;
    }

    struct entry {
        std::vector<uint8_t> raw;
        std::list<row_id>::iterator lru;
    };

    bool fetch_hit(row_id id, uint8_t * dest) {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return false;
        std::memcpy(dest, it->second.raw.data(), row_bytes_);
        lru_.splice(lru_.begin(), lru_, it->second.lru);
        ++stats_.hit;
        return true;
    }

    void insert(row_id id, const uint8_t * data) {
        if (entries_.find(id) != entries_.end()) return;
        while (entries_.size() >= row_capacity_) {
            const row_id old = lru_.back();
            entries_.erase(old);
            lru_.pop_back();
            ++stats_.evicted;
        }
        std::vector<uint8_t> raw(data, data + row_bytes_);
        lru_.push_front(id);
        try {
            entries_.emplace(id, entry{std::move(raw), lru_.begin()});
        } catch (...) {
            lru_.pop_front();
            throw;
        }
    }

    void run() noexcept {
        std::vector<uint8_t> raw; // one scratch row per worker lifetime
        for (;;) {
            row_id id = 0;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_work_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                id = queue_.front();
                queue_.pop_front();
            }
            // Serialize callback and publication with foreground readers.
            // A queued id can become resident while this worker waits for the
            // source. Recheck after taking the same lock used by get_locked.
            std::lock_guard<std::mutex> source_lock(reader_mu_);
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stopping_) return;
                if (entries_.count(id)) {
                    pending_.erase(id);
                    ++stats_.already_filled;
                    cv_ready_.notify_all();
                    continue;
                }
            }
            bool ok = false;
            try {
                raw.resize(row_bytes_);
                ok = reader_(id, raw.data(), raw.size());
            } catch (...) { ok = false; }
            {
                std::lock_guard<std::mutex> lock(mu_);
                pending_.erase(id);
                if (!stopping_) {
                    if (ok) {
                        ++stats_.background_reads;
                        try { insert(id, raw.data()); }
                        catch (...) { ++stats_.read_failures; }
                    } else {
                        ++stats_.read_failures;
                    }
                }
            }
            cv_ready_.notify_all();
        }
    }

    const reader_type reader_;
    const row_id total_rows_;
    const size_t row_bytes_;
    const size_t row_capacity_;
    const size_t queue_capacity_;
    mutable std::mutex mu_;
    std::mutex reader_mu_;
    std::mutex shutdown_mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_ready_;
    std::thread worker_;
    std::deque<row_id> queue_;
    std::unordered_set<row_id> pending_;
    std::unordered_map<row_id, entry> entries_;
    std::list<row_id> lru_;
    counters stats_;
    size_t active_reads_ = 0;
    bool stopping_ = false;
};

} // namespace llama_exp
