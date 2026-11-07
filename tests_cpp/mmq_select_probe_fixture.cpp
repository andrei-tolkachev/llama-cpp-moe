// P15C host fixture. No nvcc/GPU; output is JSONL on stderr only if opt-in.
#include "mmq_select_probe.hpp"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    const bool profile = llama_exp::mmq_select_probe::enabled();
    if (!profile) {
        std::cout << "P15C host profile disabled\n";
        return 0;
    }
    const bool overflow = argc > 1 && argv[1][0] == 'o';
    const bool saturate = argc > 1 && argv[1][0] == 's';
    auto & probe = llama_exp::mmq_select_probe::local();
    if (saturate) {
        probe.record(42, true, 1, false, std::numeric_limits<std::uint64_t>::max());
        probe.record(42, true, 1, true, 30);
        probe.flush();
        return 0;
    }
    if (overflow) {
        for (int type = 0; type < 529; ++type) {
            probe.record(type, true, 1, false, 100);
        }
        probe.flush();
        return 0;
    }
    for (int i=0; i<4096; ++i) {
        probe.record(12, true, 1, i!=0, 100); // triggers periodic flush
    }
    for (int i=0; i<32; ++i) {
        probe.record(13, false, 32, false, 200);
    }
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([t] {
            auto & p = llama_exp::mmq_select_probe::local();
            for (int i=0; i<64; ++i) {
                p.record(10+t, true, 128, i>0, 70);
            }
        });
    }
    for (auto & thread: workers) thread.join();
    probe.flush();
    std::cout << "P15C host profile enabled\n";
    return 0;
}
