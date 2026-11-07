// Direct GGML CPU dot product, packed IQ3_XXS weights (little endian).
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static uint32_t rnd(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) return 2;
    const int width = std::atoi(argv[1]), rows = std::atoi(argv[2]), passes = std::atoi(argv[3]);
    if (width <= 0 || width % 256 || rows <= 0 || passes <= 0) return 2;
    ggml_cpu_init();
    const size_t stride = ggml_row_size(GGML_TYPE_IQ3_XXS, width);
    const size_t ystride = ggml_row_size(GGML_TYPE_Q8_K, width);
    if (stride != size_t(width/256)*98 || ystride != size_t(width/256)*292) return 3;
    std::vector<uint8_t> x(rows*stride), y(ystride);
    std::vector<float> out(rows);
    uint32_t seed = 0x1badf00d;
    for (uint8_t &v : x) v = uint8_t(rnd(seed));
    for (size_t i=0; i<x.size(); i+=98) {
        const uint16_t fp16 = uint16_t(0x2800 + (rnd(seed) & 0xfff));
        std::memcpy(x.data()+i, &fp16, 2);
    }
    if (argc == 6) {
        std::ifstream packed(argv[5], std::ios::binary);
        if (!packed.read(reinterpret_cast<char *>(x.data()), x.size()) || packed.peek() != EOF) return 5;
    }
    for (uint8_t &v : y) v = uint8_t(rnd(seed));
    for (size_t i=0; i<y.size(); i+=292) {
        const float delta = float(1+(rnd(seed)%32))/128.f;
        std::memcpy(y.data()+i, &delta, 4);
        y[i+4] = 128;  // signed int8 minimum exercises wrapping negation
    }
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS)->vec_dot;
    auto run = [&] {
        for (int r=0; r<rows; ++r)
            dot(width, &out[r], 0, x.data()+size_t(r)*stride, 0, y.data(), 0, 1);
    };
    run();
    const auto begin = std::chrono::steady_clock::now();
    for (int p=0; p<passes; ++p) run();
    const auto end = std::chrono::steady_clock::now();
    std::ofstream file(argv[4], std::ios::binary);
    file.write(reinterpret_cast<const char *>(out.data()), out.size()*sizeof(float));
    std::cout << std::chrono::duration<double,std::nano>(end-begin).count()/(rows*double(passes)) << "\n";
    return file.good() ? 0 : 4;
}
