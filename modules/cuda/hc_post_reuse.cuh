#pragma once

// P15A: opt-in F32 DeepSeek HC_POST input reuse, no additional CUDA stream or
// allocation. Included by dsv4-hc.cu after common.cuh; not a public ggml API.
// The caller retains the legacy dispatch for all unsupported shapes/layouts.

#include <cstdint>
#include <limits>

static inline bool llama_exp_dsv4_hc_post_reuse_eligible(
        int64_t n_embd, int64_t hc, int64_t n_tokens, bool has_comb, bool gated) {
    // Keep short decode and tiny prefill on the existing (4x wider) kernel.
    if (!has_comb || gated || hc != 4 || n_embd < 256 || n_tokens < 8) {
        return false;
    }
    // Avoid overflow before computing the launch grid size.
    if (n_embd > std::numeric_limits<int64_t>::max() / n_tokens) {
        return false;
    }
    return n_embd*n_tokens >= 65536;
}

// One GPU thread processes a single (embedding, token) across all 4 HC
// destinations.  Compared with the legacy (embedding, HC, token) kernel,
// the source x is loaded once rather than four times and the four residual
// values are read once each rather than once per destination.
// Arbitrary element strides from the original op are preserved.
static __global__ void dsv4_hc_post_reuse_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= n_embd*n_tokens) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    const float xv = x[i0*sx0 + it*sx1];
    const float r0 = residual[i0*sr0 + 0*sr1 + it*sr2];
    const float r1 = residual[i0*sr0 + 1*sr1 + it*sr2];
    const float r2 = residual[i0*sr0 + 2*sr1 + it*sr2];
    const float r3 = residual[i0*sr0 + 3*sr1 + it*sr2];

#pragma unroll
    for (int64_t idst = 0; idst < 4; ++idst) {
        // Keep the baseline accumulation order (4 residual contributions).
        float sum = xv * post[idst*sp0 + it*sp1];
        sum += r0 * comb[idst*sc0 + 0*sc1 + it*sc2];
        sum += r1 * comb[idst*sc0 + 1*sc1 + it*sc2];
        sum += r2 * comb[idst*sc0 + 2*sc1 + it*sc2];
        sum += r3 * comb[idst*sc0 + 3*sc1 + it*sc2];
        dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
    }
}
