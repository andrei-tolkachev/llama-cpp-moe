#pragma once

// P11: persistent MoE CPU producer with fixed-capacity control metadata.
// The P10 job previously allocated a std::function + two std::vectors for
// every token/cache update. This version uses an 8-element array and a
// non-owning, type-erased function pointer to the caller's stack job frame.
// The worker MUST finish before execute() returns; backend operations stay
// on the calling thread. Only the caller's prepare() may allocate spans.
// No GGML/CUDA dependencies. C++17.

#include "upload_types.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace llama_exp {

class moe_upload_worker final {
public:
    static bool enabled() {
        static const bool value = [] {
            const char * env = std::getenv("LLAMA_EXPERIMENTAL_MOE_PERSISTENT");
            return env && env[0] == '1' && env[1] == '\0';
        }();
        return value;
    }

    moe_upload_worker() : thread_([this] { run(); }) {}
    moe_upload_worker(const moe_upload_worker &) = delete;
    moe_upload_worker & operator=(const moe_upload_worker &) = delete;

    // The owning scheduler must not destroy the worker while execute() runs.
    ~moe_upload_worker() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    template <class Prepare, class Submit, class AbortFence>
    bool execute(uint8_t * staging, size_t capacity, size_t banks, size_t minimum_slice,
                 Prepare prepare, Submit submit, AbortFence abort_fence) {
        constexpr size_t alignment = 256;
        if (!staging || banks < 2 || banks > max_banks || minimum_slice == 0 ||
                capacity < banks || capacity / banks < alignment) {
            return false;
        }
        const size_t slice_bytes = (capacity / banks / alignment)*alignment;
        if (slice_bytes < minimum_slice) return false;

        // The frame is borrowed by the worker for THIS invocation only.
        // await_done() is compulsory on both success and exception paths.
        struct job_frame {
            Prepare * callback;
            uint8_t * buffer;
            size_t slice_bytes;
        } frame{&prepare, staging, slice_bytes};
        const auto invoke = +[](void * opaque, size_t bank) -> moe_upload_batch {
            auto & f = *static_cast<job_frame *>(opaque);
            return (*f.callback)(bank, f.buffer + bank*f.slice_bytes, f.slice_bytes);
        };
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (active_ || stop_) {
                throw std::logic_error("MoE worker is active or stopping");
            }
            ready_.fill(false);
            error_ = nullptr;
            total_banks_ = banks;
            job_ctx_ = &frame;
            job_fn_ = invoke;
            cancel_ = false;
            done_ = false;
            active_ = true;
            job_ready_ = true;
        }
        cv_.notify_all();

        bool submission_attempted = false;
        try {
            // Bank zero runs on the calling scheduler thread. The worker
            // handles banks [1..N) using separate, non-overlapping slices.
            const auto first = invoke(&frame, 0);
            submission_attempted = true;  // submit() may enqueue and throw
            submit(0, first);
            for (size_t bank = 1; bank < banks; ++bank) {
                std::unique_lock<std::mutex> guard(mutex_);
                cv_.wait(guard, [&] { return ready_[bank] || error_ || done_; });
                if (error_) std::rethrow_exception(error_);
                if (!ready_[bank]) throw std::runtime_error("MoE worker incomplete upload");
                // No worker writes this bank again before execute returns.
                const moe_upload_batch * prepared = &batches_[bank];
                guard.unlock();
                submission_attempted = true;
                submit(bank, *prepared);
            }
            await_done();
        } catch (...) {
            {
                std::lock_guard<std::mutex> guard(mutex_);
                cancel_ = true;
            }
            cv_.notify_all();
            await_done();
            if (submission_attempted) {
                try { abort_fence(); } catch (...) { std::terminate(); }
            }
            clear_job();
            throw;  // never fall back to serial after a partial submission
        }
        clear_job();
        return true;
    }

private:
    static constexpr size_t max_banks = 8;
    using job_fn = moe_upload_batch (*)(void *, size_t);

    void await_done() {
        std::unique_lock<std::mutex> guard(mutex_);
        cv_.wait(guard, [&] { return done_; });
    }

    void clear_job() {
        std::lock_guard<std::mutex> guard(mutex_);
        // The worker is finished. Release payload descriptors and callbacks
        // without retaining any source/staging pointers across invocations.
        for (size_t i = 1; i < total_banks_; ++i) {
            batches_[i].spans.clear();
            batches_[i].bytes = 0;
            batches_[i].staged_bytes = 0;
            batches_[i].calls_saved = 0;
        }
        job_fn_ = nullptr;
        job_ctx_ = nullptr;
        total_banks_ = 0;
        active_ = false;
    }

    void run() {
        for (;;) {
            std::unique_lock<std::mutex> guard(mutex_);
            cv_.wait(guard, [&] { return job_ready_ || stop_; });
            if (stop_) return;
            job_ready_ = false;
            const size_t banks = total_banks_;
            const job_fn callback = job_fn_;
            void * const context = job_ctx_;
            // The borrower cannot clear the frame until done_ is true.
            guard.unlock();

            for (size_t bank = 1; bank < banks; ++bank) {
                {
                    std::lock_guard<std::mutex> state(mutex_);
                    if (cancel_) break;
                }
                try {
                    auto prepared = callback(context, bank);
                    {
                        std::lock_guard<std::mutex> state(mutex_);
                        batches_[bank] = std::move(prepared);
                        ready_[bank] = true;
                    }
                    cv_.notify_all();
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> state(mutex_);
                        error_ = std::current_exception();
                    }
                    cv_.notify_all();
                    break;
                }
            }
            {
                std::lock_guard<std::mutex> state(mutex_);
                done_ = true;
            }
            cv_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::array<moe_upload_batch, max_banks> batches_;
    std::array<bool, max_banks> ready_{};
    job_fn job_fn_ = nullptr;
    void * job_ctx_ = nullptr;
    std::exception_ptr error_;
    size_t total_banks_ = 0;
    bool active_ = false;
    bool job_ready_ = false;
    bool done_ = false;
    bool cancel_ = false;
    bool stop_ = false;
    // run() may observe every member as soon as the thread constructor starts.
    std::thread thread_;
};

} // namespace llama_exp
