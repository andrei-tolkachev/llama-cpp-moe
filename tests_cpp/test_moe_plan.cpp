// Tests the literal moe_cache_slru extracted from the patched source file.
// Only the assertion/abort macros are supplied by this standalone CPU harness.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define GGML_ASSERT(condition) do { if (!(condition)) { std::abort(); } } while (0)
#define GGML_ABORT(...) std::abort()
#include "planner_under_test.h"

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

struct state {
    moe_cache_slru cache;
    std::vector<std::vector<int32_t>> maps;
    std::vector<moe_cache_slru::fill> fills;

    state(int32_t nl, int32_t ne, int32_t ns) : maps(nl, std::vector<int32_t>(ne, -1)) {
        cache.init(nl, ne, ns);
        for (int32_t il = 0; il < nl; ++il) {
            cache.slot_map[il] = maps[il].data();
        }
    }
    state(const state &) = delete;
    state & operator=(const state &) = delete;
};

struct request {
    int32_t layer;
    std::vector<int32_t> ids;
};

static void invariants(const state & s) {
    const auto & c = s.cache;
    std::vector<bool> visited(c.n_slots, false);
    int protected_count = 0;
    for (int segment = 0; segment < 2; ++segment) {
        int32_t previous = -1;
        int count = 0;
        for (int32_t slot = c.head[segment]; slot >= 0; slot = c.next[slot]) {
            require(slot < c.n_slots && !visited[slot], "invalid/cyclic SLRU list");
            visited[slot] = true;
            require(c.prev[slot] == previous, "SLRU previous link mismatch");
            require(c.is_protected[slot] == bool(segment), "SLRU segment mismatch");
            previous = slot;
            ++count;
        }
        require(previous == c.tail[segment], "SLRU tail mismatch");
        if (segment) {
            protected_count = count;
        }
    }
    require(std::all_of(visited.begin(), visited.end(), [](bool v) { return v; }), "unlinked slot");
    require(c.n_protected == protected_count && protected_count <= c.n_slots/2, "protected count mismatch");
    for (int32_t slot = 0; slot < c.n_slots; ++slot) {
        const int32_t key = c.key_of[slot];
        if (key >= 0) {
            require(size_t(key/c.n_expert) < s.maps.size(), "invalid key layer");
            require(s.maps[key/c.n_expert][key%c.n_expert] == slot, "key/map mismatch");
        }
    }
    for (size_t il = 0; il < s.maps.size(); ++il) {
        require(c.slot_map[il] == s.maps[il].data(), "slot map pointer changed");
        for (int32_t expert = 0; expert < c.n_expert; ++expert) {
            const int32_t slot = s.maps[il][expert];
            require(slot >= -1 && slot < c.n_slots, "invalid slot index");
            if (slot >= 0) {
                require(c.key_of[slot] == int32_t(il)*c.n_expert + expert, "map/key mismatch");
            }
        }
    }
}

static void equal_state(const state & a, const state & b) {
    const auto & x = a.cache;
    const auto & y = b.cache;
    require(x.n_expert == y.n_expert && x.n_slots == y.n_slots, "geometry differs");
    require(a.maps == b.maps && x.key_of == y.key_of, "slot assignments differ");
    require(x.prev == y.prev && x.next == y.next, "LRU links differ");
    require(x.head[0] == y.head[0] && x.head[1] == y.head[1] &&
            x.tail[0] == y.tail[0] && x.tail[1] == y.tail[1], "LRU endpoints differ");
    require(x.is_protected == y.is_protected && x.n_protected == y.n_protected, "SLRU segments differ");
    require(x.seen_gen == y.seen_gen && x.seen == y.seen, "dedup generations differ");
    require(a.fills.size() == b.fills.size(), "fill count differs");
    for (size_t i = 0; i < a.fills.size(); ++i) {
        require(a.fills[i].expert == b.fills[i].expert && a.fills[i].slot == b.fills[i].slot,
                "ordered fills differ");
    }
    // uniq is internal scratch. The new path retains only misses or skips sorting
    // for all-hit requests; all cache state and observable plan outputs are checked.
}

static uint64_t compared_updates = 0;
static void compare_update(state & legacy, state & reuse, const request & r) {
    size_t hits_old = size_t(-1), hits_new = size_t(-1);
    const bool old_ok = legacy.cache.plan(r.layer, r.ids.data(), r.ids.size(), legacy.fills, hits_old);
    const bool new_ok = reuse.cache.plan_reuse(r.layer, r.ids.data(), r.ids.size(), reuse.fills, hits_new);
    require(old_ok == new_ok && hits_old == hits_new, "return value or hits differ");
    equal_state(legacy, reuse);
    invariants(legacy);
    invariants(reuse);
    if (new_ok) {
        for (int32_t e : r.ids) {
            require(reuse.maps[r.layer][e] >= 0, "selected expert was evicted");
        }
    } else {
        require(hits_new == 0 && reuse.fills.empty(), "failed plan has hit/fill output");
    }
    ++compared_updates;
}

static std::vector<int32_t> sequence(int n) {
    std::vector<int32_t> result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

static void differential_tests() {
    const int geometries[][3] = {
        {1, 1, 1}, {1, 2, 1}, {1, 7, 3}, {1, 16, 8}, {1, 128, 64}, {1, 512, 128},
        {2, 7, 1}, {2, 7, 7}, {2, 16, 24}, {4, 16, 8}, {4, 63, 64},
        {8, 128, 128}, {8, 512, 256}, {17, 63, 1024},
    };
    std::mt19937_64 rng(0xd2a7ce195ULL);
    for (const auto & geometry : geometries) {
        const int nl = geometry[0], ne = geometry[1], ns = geometry[2];
        state legacy(nl, ne, ns), reuse(nl, ne, ns);
        for (int iteration = 0; iteration < 5000; ++iteration) {
            if (iteration == 100 || iteration == 3000) {
                legacy.cache.seen_gen = reuse.cache.seen_gen = std::numeric_limits<uint32_t>::max() - 1;
            }
            request r{int32_t(rng()%nl), {}};
            const int count = int(rng()%(std::min(ne*2, 192) + 1));
            const int mode = int(rng()%6);
            if (mode == 3) {
                for (int e = 0; e < ne; ++e) {
                    if (legacy.maps[r.layer][e] >= 0) { r.ids.push_back(e); }
                }
                std::shuffle(r.ids.begin(), r.ids.end(), rng);
            } else if (mode >= 4) {
                r.ids = sequence(ne);
                std::shuffle(r.ids.begin(), r.ids.end(), rng);
                r.ids.resize(mode == 4 ? size_t(rng()%(ne + 1)) : size_t(std::min(ne, ns)));
            } else {
                const int single = int(rng()%ne);
                for (int i = 0; i < count; ++i) {
                    r.ids.push_back(mode == 1 ? single : int(rng()%(mode == 2 ? std::max(1, ne/8) : ne)));
                }
            }
            compare_update(legacy, reuse, r);
        }
    }
    // Exhaust probation, then continue into protected while each filled slot is
    // appended to probation. Repeated legacy scans must choose identical victims.
    for (int ns : {2, 4, 8, 64, 128, 256}) {
        state legacy(2, ns*2, ns), reuse(2, ns*2, ns);
        compare_update(legacy, reuse, {0, sequence(ns)});
        for (int cycle = 0; cycle < 256; ++cycle) {
            const int layer = cycle%2;
            auto hits = sequence(ns/2);
            std::shuffle(hits.begin(), hits.end(), rng);
            compare_update(legacy, reuse, {layer, hits});
            auto misses = sequence(ns);
            std::reverse(misses.begin(), misses.end());
            compare_update(legacy, reuse, {1-layer, misses});
        }
    }
    // The production fills vector is shared by groups. Vary its existing content
    // and capacity, including calls that fail before promotion or eviction.
    state a0(2, 7, 3), b0(2, 7, 3), a1(4, 79, 64), b1(4, 79, 64);
    for (int i = 0; i < 1000; ++i) {
        a0.fills.swap(a1.fills);
        b0.fills.swap(b1.fills);
        compare_update(a0, b0, {i%2, {int32_t(i%7), int32_t((i+1)%7), int32_t((i+2)%7)}});
        a0.fills.swap(a1.fills);
        b0.fills.swap(b1.fills);
        compare_update(a1, b1, {i%4, sequence(i%67)});
    }
    std::cout << "{\"type\":\"differential\",\"status\":\"PASS\",\"updates\":"
              << compared_updates << ",\"random_geometries\":14,\"generation_wrap_tested\":true}\n";
}

struct workload {
    const char * name;
    int nl, ne, ns;
    std::vector<request> warmup;
    std::vector<request> requests;
};

static std::vector<workload> workloads() {
    std::mt19937_64 rng(0xb34c4801ULL);
    std::vector<workload> result;
    for (const auto & shape : std::vector<std::pair<int, int>>{{32, 6}, {32, 10}, {128, 256}}) {
        workload w{shape.second == 6 ? "all_hit_top6" : shape.first == 32 ? "all_hit_short" : "all_hit_window",
                   1, 512, 256, {{0, sequence(shape.first)}}, {}};
        for (int k = 0; k < 128; ++k) {
            request r{0, {}};
            for (int i = 0; i < shape.second; ++i) { r.ids.push_back(int32_t(rng()%shape.first)); }
            w.requests.push_back(std::move(r));
        }
        result.push_back(std::move(w));
    }
    for (int count : {6, 10}) {
        workload w{count == 6 ? "mixed_top6" : "mixed_top10", 1, 512, 128, {{0, sequence(64)}}, {}};
        for (int k = 0; k < 512; ++k) {
            request r{0, {}};
            for (int i = 0; i < count; ++i) {
                r.ids.push_back(int32_t(i < (count*2)/3 ? rng()%64 : 64 + rng()%448));
            }
            std::shuffle(r.ids.begin(), r.ids.end(), rng);
            w.requests.push_back(std::move(r));
        }
        result.push_back(std::move(w));
    }
    workload mixed{"mixed_hit_window", 1, 512, 128, {{0, sequence(64)}}, {}};
    for (int k = 0; k < 512; ++k) {
        request r{0, {}};
        for (int i = 0; i < 48; ++i) { r.ids.push_back(int32_t(i < 32 ? rng()%64 : 64 + rng()%448)); }
        std::shuffle(r.ids.begin(), r.ids.end(), rng);
        mixed.requests.push_back(std::move(r));
    }
    result.push_back(std::move(mixed));
    workload cold{"all_miss_stream", 2, 512, 128, {}, {}};
    for (int k = 0; k < 128; ++k) {
        auto ids = sequence(128);
        std::shuffle(ids.begin(), ids.end(), rng);
        cold.requests.push_back({k%2, std::move(ids)});
    }
    result.push_back(std::move(cold));
    workload tiny{"tiny_random", 1, 16, 4, {}, {}};
    for (int k = 0; k < 256; ++k) { tiny.requests.push_back({0, {int32_t(rng()%16)}}); }
    result.push_back(std::move(tiny));
    workload churn{"shared_layers", 8, 128, 64, {}, {}};
    for (int k = 0; k < 256; ++k) {
        request r{int32_t(rng()%8), {}};
        for (int i = 0; i < 24; ++i) { r.ids.push_back(int32_t(rng()%128)); }
        churn.requests.push_back(std::move(r));
    }
    result.push_back(std::move(churn));
    workload scan{"probation_exhaustion", 2, 256, 128, {{0, sequence(128)}}, {}};
    for (int layer : {0, 1}) {
        auto hits = sequence(64), misses = sequence(128);
        std::shuffle(hits.begin(), hits.end(), rng);
        std::shuffle(misses.begin(), misses.end(), rng);
        scan.requests.push_back({layer, std::move(hits)});
        scan.requests.push_back({1-layer, std::move(misses)});
    }
    result.push_back(std::move(scan));
    return result;
}

struct measurement { double ns_per_plan; uint64_t checksum, hits, misses; };
static volatile uint64_t benchmark_sink = 0;

template<bool reuse>
static measurement measure(const workload & w, int iterations) {
    state s(w.nl, w.ne, w.ns);
    auto apply = [&](const request & r, size_t & hit) {
        if constexpr (reuse) {
            return s.cache.plan_reuse(r.layer, r.ids.data(), r.ids.size(), s.fills, hit);
        } else {
            return s.cache.plan(r.layer, r.ids.data(), r.ids.size(), s.fills, hit);
        }
    };
    size_t hit = 0;
    for (const request & r : w.warmup) { require(apply(r, hit), "warmup capacity"); }
    for (const request & r : w.requests) { require(apply(r, hit), "stream warmup capacity"); }
    uint64_t hits = 0, misses = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        const request & r = w.requests[size_t(i)%w.requests.size()];
        require(apply(r, hit), "benchmark capacity");
        hits += hit;
        misses += s.fills.size();
    }
    const auto stop = std::chrono::steady_clock::now();
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto & map : s.maps) {
        for (int32_t value : map) { checksum = (checksum ^ uint32_t(value))*1099511628211ULL; }
    }
    for (const auto * values : {&s.cache.key_of, &s.cache.prev, &s.cache.next}) {
        for (int32_t value : *values) { checksum = (checksum ^ uint32_t(value))*1099511628211ULL; }
    }
    benchmark_sink = checksum;
    return {std::chrono::duration<double, std::nano>(stop-start).count()/iterations, checksum, hits, misses};
}

static void failed_upload_recovery() {
    state s(2, 8, 4);
    const int32_t ids[] = {2, 5};
    size_t hits = 0;
    for (int layer : {0, 0, 1, 1}) {
        require(s.cache.plan(layer, ids, 2, s.fills, hits), "initial plan");
    }
    s.cache.invalidate_banks();
    invariants(s);
    for (int layer : {0, 1}) {
        require(s.cache.plan_reuse(layer, ids, 2, s.fills, hits), "retry plan");
        require(hits == 0 && s.fills.size() == 2, "partial bank reused on retry");
    }
}

int main(int argc, char ** argv) {
    try {
        failed_upload_recovery();
        differential_tests();
        if (argc > 1 && std::string(argv[1]) == "--bench") {
            const int iterations = argc > 2 ? std::stoi(argv[2]) : 100000;
            const int repeats = argc > 3 ? std::stoi(argv[3]) : 7;
            const std::string selected = argc > 4 ? std::string(",") + argv[4] + "," : "";
            int measured_cases = 0;
            require(iterations > 0 && repeats > 0, "positive benchmark dimensions required");
            for (const workload & w : workloads()) {
                if (!selected.empty() && selected.find(std::string(",") + w.name + ",") == std::string::npos) {
                    continue;
                }
                ++measured_cases;
                // Tiny plans need longer samples than window-sized plans to avoid
                // treating scheduler jitter in a 1-2 ms run as an optimization.
                const int case_iterations = std::max(iterations,
                    std::string(w.name) == "tiny_random" ? 5000000 :
                    (std::string(w.name) == "all_hit_short" || std::string(w.name) == "all_hit_top6" ||
                     std::string(w.name) == "mixed_top6" || std::string(w.name) == "mixed_top10") ? 1000000 : 1);
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    measurement old, updated;
                    if (repeat%2) { updated = measure<true>(w, case_iterations); old = measure<false>(w, case_iterations); }
                    else { old = measure<false>(w, case_iterations); updated = measure<true>(w, case_iterations); }
                    require(old.checksum == updated.checksum && old.hits == updated.hits && old.misses == updated.misses,
                            "benchmark traces disagree");
                    std::cout << "{\"type\":\"benchmark\",\"case\":\"" << w.name << "\",\"sample\":" << repeat
                              << ",\"iterations\":" << case_iterations << ",\"baseline_ns_per_plan\":" << old.ns_per_plan
                              << ",\"reuse_ns_per_plan\":" << updated.ns_per_plan << ",\"speedup\":" << old.ns_per_plan/updated.ns_per_plan
                              << ",\"hits\":" << old.hits << ",\"misses\":" << old.misses << "}\n";
                }
            }
            require(measured_cases > 0, "no benchmark case matched");
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
