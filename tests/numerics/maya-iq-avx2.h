// Adapted from Project Maya ds4_iq_cpu.cpp. See LICENSE-MAYA.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#define IQ_AVX2 __attribute__((target("avx2,fma")))
namespace maya_iq {
inline float h2f(uint16_t h) {   // F16 -> F32, exactly
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t b;
    if (e == 0) {
        const float f = (float) m * 5.9604644775390625e-8f;   // m 2^-24 (exact)
        std::memcpy(&b, &f, 4);
        b |= s;
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

IQ_AVX2 inline float hsum_ggml(__m256 x) {   // ggml's hsum_float_8, the same additions in the same order
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
// IQ2_S / IQ3_S: 32 explicit sign bits -> 32 bytes of -1 / +1 (ggml's bit selector)
IQ_AVX2 inline __m256i sgn32(uint32_t m) {
    const __m256i sh = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,   //
                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i b = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) m), sh), sel);
    return _mm256_or_si256(_mm256_cmpeq_epi8(b, sel), _mm256_set1_epi8(1));
}
// table entries into a vector without the shuffle port (ggml's _mm256_set_epi64x / _epi32 of table loads ends up in
// inserts, one shuffle each; here each entry is a broadcast load and the lanes are put together by dword blends)
IQ_AVX2 inline __m256i bq(const void* p) { return _mm256_castpd_si256(_mm256_broadcast_sd((const double*) p)); }
IQ_AVX2 inline __m256i bd(const void* p) { return _mm256_castps_si256(_mm256_broadcast_ss((const float*) p)); }
IQ_AVX2 inline __m256i tab4(const uint64_t* t, unsigned a, unsigned b, unsigned c, unsigned d) {   // qwords t[a] t[b] t[c] t[d]
    return _mm256_blend_epi32(_mm256_blend_epi32(bq(t + a), bq(t + b), 0x0C), _mm256_blend_epi32(bq(t + c), bq(t + d), 0xC0),
                              0xF0);
}
IQ_AVX2 inline __m256i tab8(const uint32_t* t, const unsigned* i) {   // dwords t[i[0]] .. t[i[7]]
    const __m256i a = _mm256_blend_epi32(bd(t + i[0]), bd(t + i[1]), 0x02), b = _mm256_blend_epi32(bd(t + i[2]), bd(t + i[3]), 0x08);
    const __m256i c = _mm256_blend_epi32(bd(t + i[4]), bd(t + i[5]), 0x20), e = _mm256_blend_epi32(bd(t + i[6]), bd(t + i[7]), 0x80);
    return _mm256_blend_epi32(_mm256_blend_epi32(a, b, 0x0C), _mm256_blend_epi32(c, e, 0xC0), 0xF0);
}
// the 8 sign bytes (-1 / +1) of each 7-bit sign index with its parity bit: ggml's keven_signs_q2xs, as +-1 bytes
alignas(64) uint64_t g_ks64[128];
void build_ks64() {
    for (int i = 0; i < 128; ++i) {
        int par = 0;
        for (int b = 0; b < 7; ++b) par ^= (i >> b) & 1;
        const int s = i | (par << 7);
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) v |= (uint64_t) (((s >> j) & 1) ? 0xFFu : 0x01u) << (8 * j);
        g_ks64[i] = v;
    }
}
IQ_AVX2 inline __m256i sgn7t(uint32_t w) {   // a chunk's four 7-bit sign indices (bits 0..27 of w)
    return tab4(g_ks64, w & 127u, (w >> 7) & 127u, (w >> 14) & 127u, (w >> 21) & 127u);
}
IQ_AVX2 inline __m256i sc16(int a, int b) {   // int16 lanes 0-7 a (values 0-15), 8-15 b (values 16-31)
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)), _mm_set1_epi16((short) b), 1);
}
bool g_gather18 = false, g_gather21 = false;   // the IQ3_XXS / IQ3_S grids by vpgatherdd (own(): if faster here)
// IQ2_XXS's grid entry with its signs applied, for every grid index g and sign index i, at g << 7 | i (256 KB, built on
// first use): one lookup instead of two; the products are then (q8 + 128) x signed grid less 128 x the grid's pair sums
// (vpmaddubsw takes u8 x s8), the same integers as ggml's |grid| x signed q8
alignas(64) uint64_t g_ct2x[256 * 128];
bool g_ct16 = false;   // (own(): if faster here)
void build_ct2x() {
    for (int g = 0; g < 256; ++g)
        for (int i = 0; i < 128; ++i) {
            const uint64_t sg = g_ks64[i];   // (+-1 bytes)
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) {
                const int gb = (int) ((iq2xxs_grid[g] >> (8 * j)) & 255u);
                const int sb = ((sg >> (8 * j)) & 255u) == 0xFFu ? -gb : gb;
                v |= (uint64_t) (uint8_t) (int8_t) sb << (8 * j);
            }
            g_ct2x[(size_t) g << 7 | (size_t) i] = v;
        }
}

// rows [r0, r1) (out[t][r - r0]) of NT tokens; TY 16 = IQ2_XXS, 22 = IQ2_S, 18 = IQ3_XXS, 21 = IQ3_S.  A pair of
// 32-value chunks a step: grid magnitudes g1 / g2, sign vectors s1 / s2 (-1 / +1), scales c1 / c2 - decoded once - then
// every token: q8 times the signs, vpmaddubsw against the grid, vpmaddwd by the scales, into ggml's 8 int32 lanes
template <int TY, int NT>
IQ_AVX2 void iq_rows_avx2(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0,
                          int r1) {
    const __m256i sub = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const bool gat = TY == 18 ? g_gather18 : TY == 21 ? g_gather21 : false;
    const bool ct = TY == 16 && g_ct16;
    const __m256i k80 = _mm256_set1_epi8((char) 0x80);
    constexpr float K = TY == 18 ? 0.25f : TY == 21 ? 1.0f : 0.125f;
    constexpr size_t BB = TY == 16 ? sizeof(block_iq2_xxs) : TY == 22 ? sizeof(block_iq2_s) :
                          TY == 18 ? sizeof(block_iq3_xxs) : sizeof(block_iq3_s);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * BB;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                __m256i g1, g2, s1, s2, c1, c2;
                if (TY == 16) {   // IQ2_XXS: per chunk a word of 4 grid indices, a word of 4 sign indices + the scale
                    const uint8_t* q = blk + 2 + 8 * ib;
                    const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
                    if (ct) {   // (g1 / g2 the signed grids; s1 / s2 their 128 x pair sums)
                        g1 = tab4(g_ct2x, (a0 & 255u) << 7 | (a1 & 127u), ((a0 >> 8) & 255u) << 7 | ((a1 >> 7) & 127u),
                                  ((a0 >> 16) & 255u) << 7 | ((a1 >> 14) & 127u), (a0 >> 24) << 7 | ((a1 >> 21) & 127u));
                        g2 = tab4(g_ct2x, (b0 & 255u) << 7 | (b1 & 127u), ((b0 >> 8) & 255u) << 7 | ((b1 >> 7) & 127u),
                                  ((b0 >> 16) & 255u) << 7 | ((b1 >> 14) & 127u), (b0 >> 24) << 7 | ((b1 >> 21) & 127u));
                        s1 = _mm256_maddubs_epi16(k80, g1);
                        s2 = _mm256_maddubs_epi16(k80, g2);
                    } else {
                        g1 = tab4(iq2xxs_grid, a0 & 255u, (a0 >> 8) & 255u, (a0 >> 16) & 255u, a0 >> 24);
                        g2 = tab4(iq2xxs_grid, b0 & 255u, (b0 >> 8) & 255u, (b0 >> 16) & 255u, b0 >> 24);
                        s1 = sgn7t(a1);
                        s2 = sgn7t(b1);
                    }
                    c1 = _mm256_set1_epi16((short) (2 * (a1 >> 28) + 1));
                    c2 = _mm256_set1_epi16((short) (2 * (b1 >> 28) + 1));
                } else if (TY == 22) {   // IQ2_S: 10-bit grid indices (qs | 2 bits of qh), explicit signs, 2 scales a chunk
                    const uint8_t* qs = blk + 2 + 4 * ib;
                    const uint8_t* qh = blk + 2 + QK_K / 4;
                    const uint8_t* sg = blk + 2 + QK_K / 8 + 4 * ib;
                    const uint8_t* scl = blk + 2 + QK_K / 4 + QK_K / 32;
                    const unsigned h0 = qh[ib], h1 = qh[ib + 1];
                    g1 = tab4(iq2s_grid, qs[0] | ((h0 << 8) & 0x300u), qs[1] | ((h0 << 6) & 0x300u), qs[2] | ((h0 << 4) & 0x300u),
                              qs[3] | ((h0 << 2) & 0x300u));
                    g2 = tab4(iq2s_grid, qs[4] | ((h1 << 8) & 0x300u), qs[5] | ((h1 << 6) & 0x300u), qs[6] | ((h1 << 4) & 0x300u),
                              qs[7] | ((h1 << 2) & 0x300u));
                    s1 = sgn32(u32(sg));
                    s2 = sgn32(u32(sg + 4));
                    c1 = sc16(2 * (scl[ib] & 0xf) + 1, 2 * (scl[ib] >> 4) + 1);
                    c2 = sc16(2 * (scl[ib + 1] & 0xf) + 1, 2 * (scl[ib + 1] >> 4) + 1);
                } else if (TY == 18) {   // IQ3_XXS: 8 grid indices a chunk, then the chunks' sign / scale words
                    const uint8_t* q3 = blk + 2 + 8 * ib;
                    const uint8_t* gas = blk + 2 + QK_K / 4 + 4 * ib;
                    if (gat) {
                        g1 = _mm256_i32gather_epi32((const int*) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q3)), 4);
                        g2 = _mm256_i32gather_epi32((const int*) iq3xxs_grid,
                                                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (q3 + 8))), 4);
                    } else {
                        unsigned i1[8], i2[8];
                        for (int k = 0; k < 8; ++k) {
                            i1[k] = q3[k];
                            i2[k] = q3[8 + k];
                        }
                        g1 = tab8(iq3xxs_grid, i1);
                        g2 = tab8(iq3xxs_grid, i2);
                    }
                    const uint32_t a0 = u32(gas), a1 = u32(gas + 4);
                    s1 = sgn7t(a0);
                    s2 = sgn7t(a1);
                    c1 = _mm256_set1_epi16((short) (2 * (a0 >> 28) + 1));
                    c2 = _mm256_set1_epi16((short) (2 * (a1 >> 28) + 1));
                } else {   // IQ3_S: 9-bit grid indices (qs | a bit of qh), explicit signs
                    const uint8_t* qs = blk + 2 + 8 * ib;
                    const uint8_t* qh = blk + 2 + QK_K / 4;
                    const uint8_t* sg = blk + 2 + QK_K / 4 + QK_K / 32 + 4 * ib;
                    const int sc = blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + ib / 2];
                    const unsigned h0 = qh[ib], h1 = qh[ib + 1];
                    if (gat) {
                        const __m256i hb1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) h0), sub), _mm256_set1_epi32(1)), 8);
                        const __m256i hb2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) h1), sub), _mm256_set1_epi32(1)), 8);
                        g1 = _mm256_i32gather_epi32((const int*) iq3s_grid,
                                                    _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs)), hb1), 4);
                        g2 = _mm256_i32gather_epi32((const int*) iq3s_grid,
                                                    _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (qs + 8))), hb2), 4);
                    } else {
                        unsigned i1[8], i2[8];
                        for (int k = 0; k < 8; ++k) {
                            i1[k] = qs[k] | (((h0 >> k) & 1u) << 8);
                            i2[k] = qs[8 + k] | (((h1 >> k) & 1u) << 8);
                        }
                        g1 = tab8(iq3s_grid, i1);
                        g2 = tab8(iq3s_grid, i2);
                    }
                    s1 = sgn32(u32(sg));
                    s2 = sgn32(u32(sg + 4));
                    c1 = _mm256_set1_epi16((short) (2 * (sc & 0xf) + 1));
                    c2 = _mm256_set1_epi16((short) (2 * (sc >> 4) + 1));
                }
                if (ct) {
                    for (int t = 0; t < NT; ++t) {
                        const int8_t* q8 = y[t][i].qs + 32 * ib;
                        const __m256i u1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*) q8), k80);   // (q8 + 128)
                        const __m256i u2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*) (q8 + 32)), k80);
                        const __m256i d1 = _mm256_sub_epi16(_mm256_maddubs_epi16(u1, g1), s1);
                        const __m256i d2 = _mm256_sub_epi16(_mm256_maddubs_epi16(u2, g2), s2);
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(d1, c1), _mm256_madd_epi16(d2, c2)));
                    }
                    continue;
                }
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + 32 * ib;
                    const __m256i d1 = _mm256_maddubs_epi16(g1, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) q8), s1));
                    const __m256i d2 = _mm256_maddubs_epi16(g2, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (q8 + 32)), s2));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(d1, c1), _mm256_madd_epi16(d2, c2)));
                }
            }
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = K * hsum_ggml(accf[t]);
    }
}


}
