#pragma once

// Exact byte reader for an immutable mmap-backed, possibly strided GGUF tensor.
// The mapping is borrowed, NEVER owned. close() the row_cache before the
// mapping is released. Stride is a byte offset between rows, not row bytes.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace llama_exp {
struct mapped_rows final {
    const uint8_t * base;
    size_t mapped_bytes;
    size_t stride;
    size_t payload_bytes;

    mapped_rows(const void * base, size_t mapped_bytes, size_t stride, size_t payload_bytes)
            : base(static_cast<const uint8_t *>(base)), mapped_bytes(mapped_bytes),
              stride(stride), payload_bytes(payload_bytes) {
        if (!base || !payload_bytes || !stride || payload_bytes > stride || payload_bytes > mapped_bytes) {
            throw std::invalid_argument("mapped rows have invalid bounds/stride");
        }
    }

    bool operator()(uint64_t id, uint8_t * dest, size_t bytes) const noexcept {
        if (!dest || bytes != payload_bytes) return false;
        // Avoid id*stride overflow before checking the mapped extent.
        if (id > (mapped_bytes - payload_bytes)/stride) return false;
        std::memcpy(dest, base + size_t(id)*stride, payload_bytes);
        return true;
    }
};
}
