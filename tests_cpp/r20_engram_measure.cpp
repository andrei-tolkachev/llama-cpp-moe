#include "engram/graph_stage.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;
using cache_type = llama_exp::engram_row_cache;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

// Exercise the interval after foreground miss lookup but before publication.
// The callback queues its own id, just as a different graph could during this read.
// Waiting for the queue to drain per iteration makes this interleaving reproducible.
static int duplicate() {
    constexpr size_t count = 512, width = 210;
    cache_type *cache_ptr = nullptr;
    const auto foreground = std::this_thread::get_id();
    std::atomic<size_t> reads{0};
    cache_type cache([&](uint64_t id, uint8_t * dest, size_t bytes) {
        ++reads;
        if (std::this_thread::get_id() == foreground)
            require(cache_ptr->prefetch(&id, 1) == 1, "enqueue");
        for (size_t j = 0; j < bytes; ++j) dest[j] = uint8_t(id + j * 15);
        return true;
    }, count, width, 8 * width, 8);
    cache_ptr = &cache;
    std::vector<uint8_t> out(width);
    const auto start = clock_type::now();
    for (size_t id = 0; id < count; ++id) {
        require(cache.get(id, out.data(), width) == cache_type::read_result::source_read, "miss");
        for (size_t j = 0; j < width; ++j) require(out[j] == uint8_t(id + j * 15), "row mismatch");
        while (cache.snapshot().queued_or_inflight) {
            require(clock_type::now() - start < std::chrono::seconds(15), "worker timeout");
            std::this_thread::yield();
        }
    }
    std::cout << "{\"rows\":" << count << ",\"reader_calls\":" << reads
              << ",\"worker_reads\":" << cache.snapshot().background_reads << "}\n";
    return 0;
}

static int stage(bool warm) {
    constexpr size_t width = 210, stride = 256, table_rows = 65536, per_layer = 32;
    const size_t groups = warm ? 50000 : 5000;
    std::vector<uint8_t> backing(table_rows * stride);
    std::mt19937 rng(754109);
    for (auto &byte : backing) byte = uint8_t(rng());
    llama_exp::mapped_rows source(backing.data(), backing.size(), stride, width);
    llama_exp::engram_graph_stage adapter(source, table_rows, 512 * width);
    std::vector<int32_t> ids(groups * per_layer);
    for (auto & id : ids) id = rng() % (warm ? 256 : table_rows);
    // Prime the 256-row hot set exactly; the cold scenario has the same setup.
    std::vector<int32_t> prime(256);
    std::iota(prime.begin(), prime.end(), 0);
    require(adapter.stage(prime.data(), prime.size()), "prime");
    const auto before = adapter.counters();
    const auto start = clock_type::now();
    uint64_t digest = 0;
    for (size_t group = 0; group < groups; ++group) {
        require(adapter.stage(ids.data() + group * per_layer, per_layer), "stage");
        digest += adapter.data()[(group * 17) % adapter.bytes()];
    }
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - start).count();
    const auto after = adapter.counters();
    // Outside the timed region, compare every output byte from 256 mixed groups.
    for (size_t group = 0; group < 256; ++group) {
        auto * requested = ids.data() + group * per_layer;
        require(adapter.stage(requested, per_layer), "verify stage");
        for (size_t r = 0; r < per_layer; ++r)
            require(std::memcmp(adapter.data() + r * width, backing.data() + requested[r] * stride, width) == 0, "bytes");
    }
    std::cout << "{\"rows\":" << ids.size() << ",\"stage_calls\":" << groups
              << ",\"wall_ns\":" << ns << ",\"digest\":" << digest
              << ",\"hits\":" << after.hit - before.hit << ",\"source_reads\":"
              << after.synchronous_reads - before.synchronous_reads << "}\n";
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "duplicate") == 0) return duplicate();
    if (argc == 2 && std::strcmp(argv[1], "warm") == 0) return stage(true);
    if (argc == 2 && std::strcmp(argv[1], "cold") == 0) return stage(false);
    return 2;
}
