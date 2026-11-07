#include "maya-iq-avx2.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <dlfcn.h>
#include <cstdlib>
#include <string>
void * new_library;

template<int TY> bool check(int n) {
    constexpr int rows = 256;
    auto type = ggml_type(TY);
    auto traits = ggml_get_type_traits_cpu(type);
    std::string symbol = std::string("ggml_vec_dot_") + ggml_type_name(type) + "_q8_K";
    auto candidate = reinterpret_cast<ggml_vec_dot_t>(dlsym(new_library, symbol.c_str()));
    if (!candidate || candidate == traits->vec_dot) { printf("invalid distinct function for %s\n", symbol.c_str()); return false; }
    size_t rb = ggml_row_size(type, n);
    std::mt19937 rng(42 + TY + n);
    std::vector<uint8_t> w(rb * rows);
    std::vector<float> x(n), ref(rows), own(rows);
    std::vector<uint8_t> aq(ggml_row_size(traits->vec_dot_type, n));
    for (auto & b : w) b = uint8_t(rng());
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < n / 256; ++block) {
            uint16_t d = ggml_fp32_to_fp16(float(int(rng() % 2048) - 1024) / 32768.f);
            memcpy(w.data() + row * rb + block * ggml_type_size(type), &d, 2);
        }
    }
    const block_q8_K * acts[] = {reinterpret_cast<const block_q8_K *>(aq.data())};
    float * outputs[] = {own.data()};
    for (int gather = 0; gather < 1; ++gather) {
        maya_iq::g_gather18 = maya_iq::g_gather21 = gather;
        for (int input = 0; input < 12; ++input) {
            for (auto & a : x) a = input == 0 ? 0.f : float(int(rng() % 65536) - 32768) / 4096.f;
            ggml_get_type_traits_cpu(traits->vec_dot_type)->from_float(x.data(), aq.data(), n);
            for (int row = 0; row < rows; ++row) traits->vec_dot(n, &ref[row], 0, w.data() + row * rb, 0, aq.data(), 0, 1);
            for (int row = 0; row < rows; ++row) candidate(n, &own[row], 0, w.data() + row * rb, 0, aq.data(), 0, 1);
            if (memcmp(ref.data(), own.data(), rows * sizeof(float))) {
                for (int row = 0; row < rows; ++row) if (memcmp(&ref[row], &own[row], 4)) {
                    printf("FAIL %s n=%d gather=%d input=%d row=%d ref=%a own=%a\n", ggml_type_name(type), n, gather, input, row, ref[row], own[row]);
                    return false;
                }
            }
        }
        std::vector<double> timings[2];
        for (int rep = 0; rep < 8; ++rep) for (int k = 0; k < 2; ++k) {
            int mode = (rep + k) % 2;
            auto begin = std::chrono::steady_clock::now();
            for (int j = 0; j < 40; ++j) {
                if (mode == 0) for (int row = 0; row < rows; ++row) traits->vec_dot(n, &ref[row], 0, w.data() + row * rb, 0, aq.data(), 0, 1);
                else for (int row = 0; row < rows; ++row) candidate(n, &own[row], 0, w.data() + row * rb, 0, aq.data(), 0, 1);
            }
            timings[mode].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
        }
        for (auto & v : timings) std::sort(v.begin(), v.end());
        printf("PASS %s n=%d gather=%d exact_rows=%d ggml_ms=%.3f candidate_ms=%.3f ratio=%.3f\n", ggml_type_name(type), n, gather, 12*rows, timings[0][4]*1000, timings[1][4]*1000, timings[0][4]/timings[1][4]);
        fflush(stdout);
    }
    return true;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    new_library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!new_library) { puts(dlerror()); return 3; }
    ggml_cpu_init();
    auto init_candidate = reinterpret_cast<void (*)()>(dlsym(new_library, "ggml_cpu_init"));
    if (!init_candidate) return 4;
    init_candidate(); maya_iq::build_ks64();
    for (int n : {256, 512, 1536, 5120, 7168}) if (!check<18>(n) || !check<21>(n) || !check<22>(n) || !check<16>(n)) return 1;
}
