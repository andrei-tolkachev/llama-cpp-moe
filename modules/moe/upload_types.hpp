#pragma once

// Common zero-copy upload descriptors for P09B and P10. Source or staging
// memory is borrowed from the caller; no backend work is done here.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_exp {

struct moe_upload_span {
    const uint8_t * data = nullptr;
    size_t dst_offset = 0;
    size_t size = 0;
};

struct moe_upload_batch {
    std::vector<moe_upload_span> spans;
    size_t bytes = 0;
    size_t staged_bytes = 0;
    size_t calls_saved = 0;
};

} // namespace llama_exp
