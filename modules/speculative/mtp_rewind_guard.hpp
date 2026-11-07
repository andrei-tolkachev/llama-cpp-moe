#pragma once

// P16B: per-sequence, bounded host snapshot of MTP's pending hidden row.
// Does NOT store or roll back KV/recurrent/Engram tensors, draft RNG, or GPU buffers.
// The upstream server restores target + draft checkpoints BEFORE calling rewind().

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace llama_exp {

inline bool mtp_rewind_guard_enabled() {
    const char * p = std::getenv("LLAMA_EXPERIMENTAL_MTP_REWIND_GUARD");
    return p != nullptr && p[0] == '1' && p[1] == '\0';
}

class mtp_rewind_guard {
public:
    // A new draft must snapshot the hidden boundary *before* draft() can
    // advance its context. The bounded storage is precisely one hidden row.
    void checkpoint(const std::vector<float> & pending_h) {
        if (pending_h.empty()) {
            throw std::invalid_argument("MTP checkpoint needs a nonempty hidden row");
        }
        checkpoint_h_ = pending_h;
        has_checkpoint_ = true;
        clear_verification();
    }

    // Used before speculative process() touches any target verification rows.
    void clear_verification() noexcept {
        verified_rows_ = 0;
        can_accept_ = false;
    }

    // Called only after source embeddings were fully copied for this sequence.
    void verified(std::size_t count) {
        if (count == 0) {
            throw std::invalid_argument("MTP verification has no token rows");
        }
        verified_rows_ = count;
        can_accept_ = true;
    }

    // Validate BEFORE modifying the live boundary. Accepting beyond the real
    // verified rows or accepting twice must fail rather than reuse stale data.
    void validate_accept(std::uint16_t count) const {
        if (!can_accept_ || verified_rows_ == 0) {
            throw std::logic_error("MTP accept without fresh verification");
        }
        if (static_cast<std::size_t>(count) >= verified_rows_) {
            throw std::out_of_range("MTP accepted more tokens than verified");
        }
    }

    void commit_accept(std::uint16_t count) {
        validate_accept(count);
        clear_verification();
        checkpoint_h_.clear();
        has_checkpoint_ = false;
    }

    // Restore only the MTP host boundary AFTER upstream restores both model
    // context snapshots. Keep the checkpoint for replay until acceptance.
    void rewind(std::vector<float> & pending_h) {
        if (!has_checkpoint_ || checkpoint_h_.empty() || checkpoint_h_.size() != pending_h.size()) {
            throw std::logic_error("MTP rewind has no matching checkpoint");
        }
        pending_h = checkpoint_h_;
        clear_verification();
    }

    // Called on request release/cancellation. Do not erase the current hidden
    // state: upstream may retain prompt KV across requests; next process()
    // is responsible for reconstructing the correct boundary.
    void discard() noexcept {
        clear_verification();
        checkpoint_h_.clear();
        has_checkpoint_ = false;
    }

    std::size_t checkpoint_bytes() const noexcept { return checkpoint_h_.size() * sizeof(float); }
    bool has_checkpoint() const noexcept { return has_checkpoint_; }
    bool can_accept() const noexcept { return can_accept_; }

private:
    std::vector<float> checkpoint_h_;
    std::size_t verified_rows_ = 0;
    bool has_checkpoint_ = false;
    bool can_accept_ = false;
};

} // namespace llama_exp
