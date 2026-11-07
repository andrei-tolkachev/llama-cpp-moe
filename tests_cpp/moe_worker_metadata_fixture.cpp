// P11 host regression: the persistent CPU worker's per-job control metadata
// must be allocation-free after construction/warm-up. No GPU operations.
#include "modules/moe/upload_worker.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>

static std::atomic<size_t> alloc_count{0};

void * operator new(std::size_t n) {
    ++alloc_count;
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](std::size_t n) {
    ++alloc_count;
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

int main() {
    using llama_exp::moe_upload_batch;
    using llama_exp::moe_upload_worker;
    constexpr size_t banks = 4;
    constexpr size_t capacity = 32768;
    constexpr size_t slice = capacity / banks;
    alignas(256) std::array<uint8_t, capacity> staging{};
    std::array<uint8_t, 4096> capture{};
    capture.fill(31);
    moe_upload_worker worker;
    const auto thread = std::this_thread::get_id();
    size_t submit_count = 0;
    size_t prepared_off_main = 0;

    for (int i = 0; i < 250; ++i) {
        size_t next_submit = 0;
        // A deliberately large closure would not fit in std::function SBO.
        // The non-owning job frame should not heap-copy it.
        auto prepare = [&, large = capture, serial = i](size_t bank, uint8_t * buf, size_t cap) -> moe_upload_batch {
            assert(large[0] == 31);
            assert(cap == slice);
            if (bank) {
                assert(std::this_thread::get_id() != thread);
                ++prepared_off_main;
            } else {
                assert(std::this_thread::get_id() == thread);
            }
            buf[0] = uint8_t((bank + serial) & 255);
            return {};
        };
        const auto submit = [&](size_t bank, const moe_upload_batch &) {
            assert(std::this_thread::get_id() == thread);
            assert(bank == next_submit++);
            assert(staging[bank*slice] == uint8_t((bank + i) & 255));
            ++submit_count;
        };
        const auto fence = [] { assert(false && "no abort should occur"); };
        if (i == 10) {
            // Nothing measured until thread/standard-library initialization
            // has been warmed up. Count only C++ new/new[] after this point.
            alloc_count.store(0);
        }
        assert(worker.execute(staging.data(), staging.size(), banks, 512, prepare, submit, fence));
        assert(next_submit == banks);
    }
    assert(submit_count == 250*banks);
    assert(prepared_off_main == 250*(banks-1));
    // The count is read before printing (iostream itself may allocate).
    const size_t counted = alloc_count.load();
#ifndef P11_SKIP_ALLOC_ASSERT
    assert(counted == 0 && "P11 worker control path allocated after warm-up");
#endif
    std::cout << "PASS: P11 reusable worker; 250 updates, large callback, FIFO, "
              << "allocation_count=" << counted << "\n";
}
