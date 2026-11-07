#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>

namespace llama_exp {

// Strata native_gu_rows groups gate/up and activation. Use GGML dots and layout here.
inline void cpu_gate_up_clamp(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * x = dst->src[0];
    const auto * ids = dst->src[1];
    const auto * gate = dst->src[2];
    const auto * up = dst->src[3];
    const auto * gt = ggml_get_type_traits_cpu(gate->type);
    const auto * ut = ggml_get_type_traits_cpu(up->type);
    const float limit = *static_cast<const float *>(userdata);
    const int64_t rows = dst->ne[0];
    int64_t active = 0;
    for (int64_t k = 0; k < ids->ne[0]; ++k) {
        const int32_t e = *reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) + k*ids->nb[0]);
        if (e >= 0) ++active;
    }
    const int64_t first = active*rows*ith/nth;
    const int64_t last = active*rows*(ith + 1)/nth;
    int64_t offset = 0;
    for (int64_t k = 0; k < ids->ne[0]; ++k) {
        const int32_t e = *reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) + k*ids->nb[0]);
        float * out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + k*dst->nb[1]);
        if (e == -1) {
            if (k%nth == ith) std::fill_n(out, rows, 0.0f);
            continue;
        }
        GGML_ASSERT(e >= 0 && e < gate->ne[2]);
        const void * q = static_cast<const char *>(x->data) + (k%x->ne[1])*x->nb[1];
        const int64_t begin = std::max(int64_t(0), first - offset);
        const int64_t end = std::min(rows, last - offset);
        offset += rows;
        for (int64_t row = begin; row < end; ++row) {
            float g, u;
            gt->vec_dot(gate->ne[0], &g, 0, static_cast<const char *>(gate->data) + e*gate->nb[2] + row*gate->nb[1], 0, q, 0, 1);
            ut->vec_dot(up->ne[0], &u, 0, static_cast<const char *>(up->data) + e*up->nb[2] + row*up->nb[1], 0, q, 0, 1);
            g = std::min(g, limit);
            u = std::clamp(u, -limit, limit);
            out[row] = g/(1.0f + expf(-g))*u;
        }
    }
}

inline ggml_tensor * build_cpu_gate_up_clamp(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                                            ggml_tensor * gate, ggml_tensor * up, const float * limit) {
    if (!gate || !up || !limit || !(*limit > 1e-6f) || !std::isfinite(*limit) ||
        x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        x->ne[2] != 1 || x->ne[3] != 1 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        !ggml_is_contiguous(x) || !ggml_is_quantized(gate->type) || !ggml_is_quantized(up->type) ||
        gate->type == GGML_TYPE_NVFP4 || up->type == GGML_TYPE_NVFP4 ||
        gate->ne[0] != x->ne[0] || up->ne[0] != x->ne[0] || gate->ne[1] != up->ne[1] ||
        gate->ne[2] != up->ne[2] || gate->ne[3] != 1 || up->ne[3] != 1 ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(up)) return nullptr;
    const auto * gt = ggml_get_type_traits_cpu(gate->type);
    const auto * ut = ggml_get_type_traits_cpu(up->type);
    if (!gt->vec_dot || !ut->vec_dot || gt->vec_dot_type != ut->vec_dot_type ||
        !ggml_get_type_traits_cpu(gt->vec_dot_type)->from_float) return nullptr;
    auto * quant = ggml_cast(ctx, x, gt->vec_dot_type);
    ggml_set_name(quant, "cpu_moe.shared_quant");
    ggml_tensor * args[] = { quant, ids, gate, up };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, gate->ne[1], ids->ne[0], 1, 1, args, 4,
                          cpu_gate_up_clamp, GGML_N_TASKS_MAX, const_cast<float *>(limit));
}

} // namespace llama_exp
