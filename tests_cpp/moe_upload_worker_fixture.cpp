#include "modules/moe/upload_worker.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llama_exp;

int main() {
    // Repeated cold starts exercise constructor/teardown before any job is submitted.
    for (size_t cycle = 0; cycle < 128; ++cycle) {
        moe_upload_worker idle;
    }
    constexpr size_t banks = 4;
    constexpr size_t row = 256;
    std::vector<uint8_t> src(banks * 2 * row);
    std::vector<uint8_t> staging(4096);
    std::vector<std::vector<uint8_t>> gpu(banks, std::vector<uint8_t>(2 * row));
    std::thread::id main_id = std::this_thread::get_id();
    std::thread::id worker_id;
    std::atomic<unsigned> background{0};
    std::vector<std::pair<size_t, moe_upload_span>> outstanding;
    int fences = 0;
    {
        moe_upload_worker worker;
        // Multiple updates must reuse exactly one producer thread.
        for (size_t generation = 1; generation < 50; ++generation) {
            for (size_t i = 0; i < src.size(); ++i) src[i] = uint8_t((i + generation * 19) & 255);
            assert(outstanding.empty());
            auto prepare = [&](size_t bank, uint8_t * dest, size_t bytes) -> moe_upload_batch {
                assert(bytes >= 512);
                if (bank > 0) {
                    ++background;
                    if (worker_id == std::thread::id{}) worker_id = std::this_thread::get_id();
                    assert(worker_id == std::this_thread::get_id());
                    assert(worker_id != main_id);
                } else { assert(std::this_thread::get_id() == main_id); }
                std::memcpy(dest, src.data() + bank * 2 * row, 2 * row);
                moe_upload_batch batch;
                batch.spans.push_back({dest, 0, 2 * row});
                batch.bytes = 2 * row;
                return batch;
            };
            auto submit = [&](size_t bank, const moe_upload_batch & batch) {
                assert(std::this_thread::get_id() == main_id);
                for (auto & span : batch.spans) outstanding.push_back({bank, span});
            };
            auto fence = [&] {
                fences++;
                for (auto & x : outstanding) {
                    assert(x.second.dst_offset + x.second.size <= gpu[x.first].size());
                    std::memcpy(gpu[x.first].data() + x.second.dst_offset, x.second.data, x.second.size);
                }
                outstanding.clear();
            };
            assert(worker.execute(staging.data(), staging.size(), banks, row, prepare, submit, fence));
            // Asynchronous device reads occur only after execute() returns.
            fence();
            for (size_t b = 0; b < banks; ++b)
                assert(std::equal(gpu[b].begin(), gpu[b].end(), src.begin() + b * 2 * row));
        }
        assert(background == 49 * (banks - 1));
        assert(fences == 49);
        // False means no job and no callback started.
        bool invoked = false;
        auto no_prepare = [&](size_t, uint8_t *, size_t) -> moe_upload_batch { invoked = true; return {}; };
        assert(!worker.execute(staging.data(), 255, 4, row, no_prepare, [](size_t, const moe_upload_batch &) {}, []{}));
        assert(!invoked);
        // Reentrant use on the same caller thread must fail closed. Running
        // the legacy serial path here would race the active staging owner.
        int rejected = 0;
        assert(worker.execute(staging.data(), staging.size(), banks, row,
            [&](size_t b, uint8_t * dest, size_t) -> moe_upload_batch {
                if (b == 0) {
                    try {
                        worker.execute(staging.data(), staging.size(), banks, row,
                            [](size_t, uint8_t *, size_t) -> moe_upload_batch { return {}; },
                            [](size_t, const moe_upload_batch &) {}, [] {});
                        assert(false);
                    } catch (const std::logic_error &) { ++rejected; }
                }
                std::memcpy(dest, src.data() + b * 2 * row, 512);
                moe_upload_batch x; x.spans.push_back({dest, 0, 512}); return x;
            }, [&](size_t b, const moe_upload_batch & x) { outstanding.push_back({b, x.spans[0]}); },
            [&] { assert(false); }));
        assert(rejected == 1);
        outstanding.clear();
        // Failure in background producer after first submit must synchronize.
        outstanding.clear();
        int before = fences;
        try {
            worker.execute(staging.data(), staging.size(), banks, row,
                [&](size_t bank, uint8_t * dest, size_t) -> moe_upload_batch {
                    if (bank == 2) throw std::runtime_error("injected prepare failure");
                    std::memcpy(dest, src.data() + bank * 2 * row, 2 * row);
                    moe_upload_batch x; x.spans.push_back({dest, 0, 512}); return x;
                }, [&](size_t bank, const moe_upload_batch & x) { outstanding.push_back({bank, x.spans[0]}); },
                [&] { fences++; outstanding.clear(); });
            assert(false);
        } catch (const std::runtime_error &) { assert(fences == before + 1); }
        // Worker survives producer failure and can accept a new update.
        outstanding.clear();
        assert(worker.execute(staging.data(), staging.size(), banks, row,
            [&](size_t bank, uint8_t * dest, size_t) -> moe_upload_batch {
                std::memcpy(dest, src.data() + bank * 2 * row, 512);
                moe_upload_batch x; x.spans.push_back({dest, 0, 512}); return x;
            }, [&](size_t b, const moe_upload_batch & x) { outstanding.push_back({b, x.spans[0]}); },
            [&] { assert(false); }));
        outstanding.clear();
        // Failure in first submit also fences (it may enqueue before throwing).
        int aborted = 0;
        try {
            worker.execute(staging.data(), staging.size(), banks, row,
                [&](size_t b, uint8_t * dest, size_t) -> moe_upload_batch {
                    std::memcpy(dest, src.data() + b * 2 * row, 512);
                    moe_upload_batch x; x.spans.push_back({dest, 0, 512}); return x;
                }, [&](size_t, const moe_upload_batch &) { throw std::runtime_error("submit failure"); },
                [&] { aborted++; });
            assert(false);
        } catch (const std::runtime_error &) { assert(aborted == 1); }
    }
    std::cout << "PASS: worker persists, 49 generations, exact deferred H2D bytes, "
                 "disabled no-op, producer failure, retry, submit abort/fence\n";
}
