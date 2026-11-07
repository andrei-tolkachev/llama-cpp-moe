#pragma once

// P08: exact-raw-row adapter for an opt-in GGML_GET_ROWS staging input.
// The caller owns the immutable mapped GGUF tensor. Do not outlive it.
// P13 optionally shares a model-owned source and async worker between graphs.
// The model must close the cache *before* releasing the tensor mapping.
// This header has no dependency on ggml; the patch to deepseek41.cpp provides
// the GGML input tensors and preserves their original quantized element type.

#include "model_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llama_exp {

class engram_graph_stage final {
public:
    static constexpr size_t max_stage_bytes = size_t(64) << 20;
    static constexpr size_t default_cache_bytes = size_t(16) << 20;

    static bool enabled() noexcept {
        // Graph reuse requires the switch to be fixed for the process lifetime.
        static const bool active = [] {
            const char * value = std::getenv("LLAMA_EXPERIMENTAL_ENGRAM_STAGE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        return active;
    }

    // Size gate is checked before any GGML tensors are replaced. Failure means
    // the original, directly mmap-backed GET_ROWS path is selected instead.
    static bool eligible(uint64_t table_rows, size_t raw_row_bytes, size_t stride,
                         size_t mapped_bytes, uint64_t rows_per_layer,
                         uint64_t number_of_layers) noexcept {
        if (!table_rows || !raw_row_bytes || !stride || raw_row_bytes > stride ||
                raw_row_bytes > (size_t(1) << 20) || mapped_bytes < raw_row_bytes ||
                rows_per_layer == 0 || number_of_layers == 0 ||
                rows_per_layer > uint64_t(INT32_MAX) ||
                table_rows - 1 > (mapped_bytes - raw_row_bytes)/stride) {
            return false;
        }
        const size_t max_rows = max_stage_bytes / raw_row_bytes;
        // Total bytes of all graph input tensors, not merely a per-layer limit.
        return rows_per_layer <= max_rows && number_of_layers <= max_rows/rows_per_layer;
    }

    // Legacy P08 constructor remains synchronous, self-contained and exact.
    engram_graph_stage(mapped_rows view, uint64_t total_rows, size_t data_budget = default_cache_bytes)
        : engram_graph_stage(std::make_shared<engram_model_cache>(view, total_rows, data_budget, 0)) {}

    // P13: the model (not the graph) owns the shared cache and worker.
    explicit engram_graph_stage(std::shared_ptr<engram_model_cache> model_cache)
        : model_cache_(std::move(model_cache)),
          total_rows_(model_cache_ ? model_cache_->total_rows() : 0),
          row_bytes_(model_cache_ ? model_cache_->row_bytes() : 0) {
        if (!model_cache_ || !row_bytes_ || !total_rows_) {
            throw std::invalid_argument("Engram graph stage needs a live model cache");
        }
    }

    engram_graph_stage(const engram_graph_stage &) = delete;
    engram_graph_stage & operator=(const engram_graph_stage &) = delete;

    // Stage raw GGUF rows, in the original index order. The GGML graph uses
    // identity indices [0, count) into a compact tensor of the SAME GGML type.
    // No conversion, dequantization, approximation, or dropped row occurs here.
    // The caller must not publish data()/bytes() if this method returns false.
    // On failure, the whole request must fail closed; switching graph sources
    // after build would be unsafe. Disable the flag for the baseline fallback.
    bool stage(const int32_t * indices, size_t count) {
        if (!indices || count == 0 || count > size_t(INT32_MAX) ||
                count > max_stage_bytes/row_bytes_) return false;
        // Validate all indices before potentially modifying the staging buffer.
        for (size_t i = 0; i < count; ++i) {
            if (indices[i] < 0 || uint64_t(indices[i]) >= total_rows_) return false;
        }
        scratch_.resize(count*row_bytes_);
        return model_cache_->get_many(indices, count, scratch_.data(), scratch_.size());
    }

    const uint8_t * data() const noexcept { return scratch_.data(); }
    size_t bytes() const noexcept { return scratch_.size(); }
    size_t row_bytes() const noexcept { return row_bytes_; }
    engram_row_cache::counters counters() const { return model_cache_->counters(); }
    size_t prefetch(const int32_t * indices, size_t count) {
        return model_cache_->prefetch(indices, count);
    }

private:
    const std::shared_ptr<engram_model_cache> model_cache_;
    const uint64_t total_rows_;
    const size_t row_bytes_;
    std::vector<uint8_t> scratch_;
};

} // namespace llama_exp
