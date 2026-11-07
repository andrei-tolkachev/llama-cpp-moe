#pragma once

// P14B: host-side completion/ownership contract for staged MoE uploads.
// Uses ONLY the existing backend's command queue; does not create a CUDA
// stream, poll device memory, change routing or time GPU kernels.
//
// An event, when available, is recorded AFTER the final slot-map upload.
// The next readback's existing backend synchronize retires the epoch. If an
// error interrupts the submission, or the owning context closes the cache
// with an epoch pending, the guard waits for all enqueued reads of borrowed
// host memory before that memory can be freed.
//
// BACKEND LIFETIME: the llama_context owner MUST call close() BEFORE freeing
// the backend. This guard deliberately never dereferences a backend in its
// destructor; an unclosed event guard terminates rather than risking UAF.

#include <cstdlib>
#include <exception>
#include <stdexcept>

namespace llama_exp {

template<class Adapter>
class moe_upload_completion final {
public:
    using backend_type = typename Adapter::backend_type;
    using event_type   = typename Adapter::event_type;

    static bool enabled() {
        static const bool on = [] {
            const char * v = std::getenv("LLAMA_EXPERIMENTAL_MOE_EVENT_FENCE");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return on;
    }

    explicit moe_upload_completion(backend_type backend) : backend_(backend) {
        if (!backend_) throw std::invalid_argument("MoE event fence: null backend");
        // nullptr means the device has no events; fall back to full backend sync
        // only for abort/close, not for every successful token.
        event_ = Adapter::create(backend_);
    }

    moe_upload_completion(const moe_upload_completion &) = delete;
    moe_upload_completion & operator=(const moe_upload_completion &) = delete;
    moe_upload_completion(moe_upload_completion &&) = delete;
    moe_upload_completion & operator=(moe_upload_completion &&) = delete;

    ~moe_upload_completion() noexcept {
        // A forgotten owner close is a programming error. We cannot safely
        // call any backend API from here: backend pointers may be dead.
        if (!closed_) std::terminate();
    }

    bool event_supported() const noexcept { return event_ != event_type{}; }
    bool pending()         const noexcept { return pending_; }
    bool closed()          const noexcept { return closed_; }

    // Must follow a backend sync (the original routing-ID readback, without
    // reordering). Can retire even if no new event has been recorded yet.
    bool retired_after_backend_sync() {
        if (closed_) throw std::logic_error("MoE event fence: retired after close");
        const bool was_pending = pending_;
        pending_  = false;
        recorded_ = false;
        return was_pending;
    }

    // Called BEFORE the first async upload, on the normal scheduler thread.
    // Once called, any exceptional exit must drain() on the same backend.
    void begin_submission() {
        if (closed_ || pending_)
            throw std::logic_error("MoE event fence: overlapping or closed epoch");
        pending_ = true;
        recorded_ = false;
    }

    // Called AFTER every bank transfer and the slot-map write have been
    // enqueued, on the SAME backend/stream (never from the CPU producer).
    void record_after_slot_map() {
        if (closed_ || !pending_ || recorded_)
            throw std::logic_error("MoE event fence: invalid record transition");
        if (event_supported()) {
            Adapter::record(event_, backend_);
            recorded_ = true;
        }
        // An eventless backend retains pending until the readback sync.
    }

    void drain() {
        if (closed_) return;
        if (!pending_) return;
        if (recorded_ && event_supported()) Adapter::synchronize_event(event_);
        else Adapter::synchronize_backend(backend_);
        pending_  = false;
        recorded_ = false;
    }

    void drain_noexcept() noexcept {
        try { drain(); } catch (...) { std::terminate(); }
    }

    // Must be invoked explicitly by llama_context before backend teardown.
    // Reentrant: close() twice is harmless; event free runs exactly once.
    void close() {
        if (closed_) return;
        drain();
        if (event_supported()) {
            Adapter::destroy(event_);
            event_ = event_type{};
        }
        closed_ = true;
    }

private:
    backend_type backend_;
    event_type event_{};
    bool pending_ = false;
    bool recorded_ = false;
    bool closed_ = false;
};

// Scope whose destructor drains on ANY exception or early exit after begin().
// The caller explicitly commit()s only after recording the last slot-map
// transfer. An abort never attempts to continue with half-uploaded banks.
template<class Completion>
class moe_upload_abort_scope final {
public:
    explicit moe_upload_abort_scope(Completion * completion) : completion_(completion) {}
    moe_upload_abort_scope(const moe_upload_abort_scope &) = delete;
    moe_upload_abort_scope & operator=(const moe_upload_abort_scope &) = delete;
    ~moe_upload_abort_scope() noexcept {
        if (completion_ && !committed_) completion_->drain_noexcept();
    }
    void commit() noexcept { committed_ = true; }
private:
    Completion * completion_;
    bool committed_ = false;
};

} // namespace llama_exp
