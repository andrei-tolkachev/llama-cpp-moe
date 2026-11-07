#pragma once

// P09: opt-in host-side MoE upload producer, C++17, no ggml dependency.
// Dependencies: the caller has already completed GPU routing ID readback and
// SLRU planning, and owns an immutable list of selected exact expert IDs.
// This class never calls a backend from its producer thread. It only copies
// CPU source bytes into disjoint slices of the caller-owned staging buffer.
// The owner MUST keep that buffer and all weight sources alive until the next
// backend synchronization, including when an upload is asynchronous.

#include "upload_types.hpp"
#include "upload_worker.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace llama_exp {

class moe_upload_pipeline final {
public:
    static bool enabled() {
        // Exact opt-in; no interpretation of arbitrary nonzero strings.
        static const bool value = [] {
            const char * env = std::getenv("LLAMA_EXPERIMENTAL_MOE_PIPELINE");
            return env && env[0] == '1' && env[1] == '\0';
        }();
        return value;
    }

    // A true return means every bank was prepared and handed to submit in
    // original bank order. A false return means the caller must use its
    // existing serial path; no preparation or submission has taken place.
    // Failures after work has started propagate and must abort the inference
    // operation (never silently continue with incomplete GPU bank contents).
    //
    // prepare(bank, slice_base, slice_bytes) is called concurrently for bank 0
    // (caller) and banks 1..n-1 (one producer). It must neither mutate the
    // SLRU/fills nor issue GPU calls. A completed batch owns descriptors only;
    // all source/staging memory stays under the caller's lifetime contract.
    // submit(bank, batch) is called ONLY on the caller thread, in bank order.
    // On a failure after submit begins, abort_fence() must synchronously drain
    // queued device reads before this function propagates the exception.
    template <class Prepare, class Submit, class AbortFence>
    static bool execute(
            uint8_t * staging, size_t capacity, size_t banks, size_t minimum_slice,
            Prepare prepare, Submit submit, AbortFence abort_fence, bool prepare_inline = false) {
        constexpr size_t alignment = 256;
        constexpr size_t max_banks = 8;
        if (!staging || banks < 2 || banks > max_banks || minimum_slice == 0 ||
                capacity < banks || capacity / banks < alignment) {
            return false;
        }
        const size_t slice_bytes = (capacity / banks / alignment) * alignment;
        if (slice_bytes < minimum_slice) {
            return false;
        }

        // No bank can pack bytes for this exact fill geometry. Preserve the
        // same slices, callbacks, bank order and abort fence, without a worker.
        if (prepare_inline) {
            bool submission_started = false;
            try {
                for (size_t i = 0; i < banks; ++i) {
                    const auto batch = prepare(i, staging + i*slice_bytes, slice_bytes);
                    submission_started = true;
                    submit(i, batch);
                }
            } catch (...) {
                if (submission_started) {
                    try { abort_fence(); } catch (...) { std::terminate(); }
                }
                throw;
            }
            return true;
        }

        // P10: reuse one producer per invoking scheduler thread. A thread-local
        // worker never holds model/weight pointers between execute() calls and
        // is safe when a scheduler processes multiple GPU backends serially.
        // Failure to create the worker falls back to the P09B per-call thread.
        if (moe_upload_worker::enabled()) {
            static thread_local std::unique_ptr<moe_upload_worker> persistent;
            static thread_local bool unavailable = false;
            if (!persistent && !unavailable) {
                try {
                    persistent = std::make_unique<moe_upload_worker>();
                } catch (const std::system_error &) {
                    unavailable = true;
                }
            }
            if (persistent) {
                return persistent->execute(staging, capacity, banks, minimum_slice, prepare, submit, abort_fence);
            }
        }
        std::vector<moe_upload_batch> batches(banks);
        std::vector<bool> ready(banks, false);
        std::mutex lock;
        std::condition_variable wake;
        std::exception_ptr producer_error;

        std::thread producer;
        try {
            producer = std::thread([&] {
                for (size_t i = 1; i < banks; ++i) {
                    try {
                        batches[i] = prepare(i, staging + i*slice_bytes, slice_bytes);
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> guard(lock);
                            producer_error = std::current_exception();
                        }
                        wake.notify_all();
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> guard(lock);
                        ready[i] = true;
                    }
                    wake.notify_all();
                }
            });
        } catch (const std::system_error &) {
            return false;
        }

        // Always join before leaving this scope, including exceptions from
        // prepare(0) or submit. No detached work can access mappings/buffers.
        struct producer_join {
            std::thread & thread;
            ~producer_join() { if (thread.joinable()) thread.join(); }
        } join{producer};

        bool submission_started = false;
        try {
            batches[0] = prepare(0, staging, slice_bytes);
            submission_started = true;
            submit(0, batches[0]);
            for (size_t i = 1; i < banks; ++i) {
                {
                    std::unique_lock<std::mutex> guard(lock);
                    wake.wait(guard, [&] { return ready[i] || producer_error; });
                    if (producer_error) {
                        std::rethrow_exception(producer_error);
                    }
                }
                submit(i, batches[i]);
            }
        } catch (...) {
            if (submission_started) {
                // A backend may already be reading staging asynchronously.
                // Its owner provides the backend-specific synchronization.
                // A failed fence cannot safely continue unwinding/freeing.
                try { abort_fence(); } catch (...) { std::terminate(); }
            }
            throw;
        }
        return true;
    }
};

} // namespace llama_exp
