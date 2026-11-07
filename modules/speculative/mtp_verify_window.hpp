#pragma once

// P16A: exact, bounded host snapshot for a single MTP speculator.
// This is deliberately independent of ggml / CUDA and does NOT roll back KV,
// recurrent state, samplers, or the target/draft contexts.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace llama_exp {

inline bool mtp_verify_window_enabled() {
    const char * value = std::getenv("LLAMA_EXPERIMENTAL_MTP_VERIFY_WINDOW");
    return value && value[0] == '1' && value[1] == '\0';
}

// The normal server batch keeps each sequence contiguous. Custom callers can
// interleave sequences; only that case needs explicit per-token indices.
class mtp_batch_rows {
public:
    template <typename Token>
    bool index(const std::vector<Token> & tokens, std::vector<int32_t> & first,
               std::vector<int32_t> & last) {
        rows_.clear();
        previous_.clear();
        if (first.size() != last.size()) {
            return false;
        }
        std::fill(first.begin(), first.end(), -1);
        std::fill(last.begin(), last.end(), -1);
        bool interleaved = false;
        for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
            const auto seq = tokens[i].seq_id;
            if (seq < 0 || static_cast<std::size_t>(seq) >= first.size()) {
                return false;
            }
            if (first[seq] < 0) {
                first[seq] = i;
            } else if (last[seq] != i - 1) {
                interleaved = true;
            }
            last[seq] = i;
        }
        if (interleaved) {
            rows_.resize(first.size());
            previous_.resize(tokens.size());
            for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
                auto & rows = rows_[tokens[i].seq_id];
                previous_[i] = rows.empty() ? -1 : rows.back();
                rows.push_back(i);
            }
        }
        return true;
    }

    int32_t previous(int32_t row) const {
        return previous_.empty() ? row - 1 : previous_[row];
    }

    int32_t count(int32_t seq, int32_t first, int32_t last) const {
        return rows_.empty() ? last - first + 1 : static_cast<int32_t>(rows_[seq].size());
    }

    int32_t at(int32_t seq, int32_t first, int32_t index) const {
        return rows_.empty() ? first + index : rows_[seq][index];
    }

private:
    std::vector<std::vector<int32_t>> rows_;
    std::vector<int32_t> previous_;
};

struct mtp_verify_window {
    std::size_t total_rows = 0;     // target verification rows, prior to compaction
    std::size_t retained_rows = 0;  // row 0..retained_rows-1 are retained

    mtp_verify_window(std::size_t n_rows, int n_max, bool single_mtp_enabled)
            : total_rows(n_rows), retained_rows(n_rows) {
        // An unknown/negative draft cap cannot safely shrink the snapshot.
        // The pinned MTP driver checks n_max after emitting its first token,
        // so even an explicit zero can produce one accepted draft token.
        if (single_mtp_enabled && n_max >= 0 && n_rows > 0) {
            const std::size_t cap = static_cast<std::size_t>(std::max(1, n_max)) + 1;
            retained_rows = std::min(n_rows, cap);
        }
    }

    std::size_t accepted_row(std::uint16_t n_accepted) const {
        if (total_rows == 0) {
            throw std::out_of_range("MTP acceptance without verified rows");
        }
        // Exact upstream selection: min(accepted, n_rows - 1).
        const std::size_t row = std::min<std::size_t>(n_accepted, total_rows - 1);
        if (row >= retained_rows) {
            // Refuse an invalid acceptance count rather than silently use a
            // wrong hidden-state row. Disable the experiment for this driver.
            throw std::out_of_range("MTP acceptance exceeds retained verify window");
        }
        return row;
    }
};

} // namespace llama_exp
