#include "moe/upload_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::cerr << "FAILED at " << __LINE__ << ": " #x "\n"; std::abort(); } } while (0)
using llama_exp::moe_upload_batch;
using llama_exp::moe_upload_pipeline;
using llama_exp::moe_upload_span;

static void test_exact_order_and_disjoint_slices() {
    constexpr size_t banks = 3;
    constexpr size_t nfill = 30;
    constexpr size_t width = 64;
    std::vector<uint8_t> source(banks*nfill*width);
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t((i * 97 + 23) % 251);
    std::vector<uint8_t> stage(12288, 0xf1);
    std::vector<uint8_t> reference(source.size());
    std::vector<uint8_t> device(source.size(), 0);
    std::atomic<int> active{0};
    std::atomic<bool> concurrent{false};
    std::atomic<bool> off_main{false};
    const auto main = std::this_thread::get_id();
    std::vector<size_t> order;
    bool used = moe_upload_pipeline::execute(stage.data(), stage.size(), banks, nfill*width,
        [&](size_t bank, uint8_t * dest, size_t cap) -> moe_upload_batch {
            CHECK(cap >= width*nfill);
            if (std::this_thread::get_id() != main) off_main = true;
            if (++active == 2) concurrent = true;
            // Indexed, non-consecutive expert positions; exact order matters.
            moe_upload_batch result;
            std::memcpy(dest, source.data()+bank*nfill*width, nfill*width);
            result.spans.push_back({dest, bank*nfill*width, nfill*width});
            result.bytes = result.staged_bytes = nfill*width;
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            --active;
            return result;
        },
        [&](size_t bank, const moe_upload_batch & data) {
            CHECK(std::this_thread::get_id() == main);
            CHECK(bank == order.size());
            order.push_back(bank);
            CHECK(data.spans.size() == 1);
            const auto & t = data.spans[0];
            std::memcpy(device.data()+t.dst_offset, t.data, t.size);
        }, [] {CHECK(false);});
    CHECK(used && concurrent && off_main && device == source);
    CHECK(order == (std::vector<size_t>{0,1,2}));
}

static void test_no_torn_async_reads() {
    // Simulate backend enqueue that does not read source until AFTER execute().
    // No staging slice may be reused within this invocation.
    std::vector<uint8_t> staging(8192, 0x4a);
    std::vector<uint8_t> weights(4*1024);
    for (size_t i=0; i<weights.size(); ++i) weights[i] = uint8_t(i*31 + 8);
    struct span { const uint8_t * data; size_t dest; size_t n; };
    std::vector<span> inflight;
    const auto success = moe_upload_pipeline::execute(staging.data(), staging.size(), 4, 512,
        [&](size_t bank, uint8_t * ptr, size_t capacity) {
            CHECK(capacity >= 1024);
            std::memcpy(ptr, weights.data()+bank*1024, 1024);
            moe_upload_batch b; b.spans.push_back({ptr,bank*1024,1024});return b;
        },
        [&](size_t, const moe_upload_batch & b) {
            for (const auto & v:b.spans) inflight.push_back({v.data,v.dst_offset,v.size});
        }, [] {CHECK(false);});
    CHECK(success && inflight.size() == 4);
    std::vector<uint8_t> device(weights.size());
    for (const auto & f:inflight) std::memcpy(device.data()+f.dest, f.data, f.n);
    CHECK(device == weights);
}

static void test_rejection_without_side_effects() {
    std::vector<uint8_t> stage(1024, 0);
    size_t calls = 0;
    auto prepare=[&](size_t, uint8_t *, size_t) {++calls;return moe_upload_batch{};};
    auto submit=[&](size_t, const moe_upload_batch &) {++calls;};
    auto fence = [] {CHECK(false);};
    CHECK(!moe_upload_pipeline::execute(nullptr, stage.size(), 3, 1, prepare, submit, fence));
    CHECK(!moe_upload_pipeline::execute(stage.data(), stage.size(), 1, 1, prepare, submit, fence));
    CHECK(!moe_upload_pipeline::execute(stage.data(), stage.size(), 9, 1, prepare, submit, fence));
    CHECK(!moe_upload_pipeline::execute(stage.data(), stage.size(), 3, 1025, prepare, submit, fence));
    CHECK(calls==0);
    CHECK(std::all_of(stage.begin(),stage.end(),[](uint8_t x){return x==0;}));
}

static void test_worker_exception_joins_and_aborts() {
    std::vector<uint8_t> stage(8192);
    std::atomic<int> active{0};
    bool caught=false;
    int fences=0;
    try {
        moe_upload_pipeline::execute(stage.data(),stage.size(),3,1024,
            [&](size_t bank,uint8_t * ptr,size_t cap) -> moe_upload_batch {
                CHECK(cap >= 1024);
                ++active;
                struct dec {std::atomic<int>& count; ~dec(){--count;}} guard{active};
                if(bank==1) throw std::runtime_error("planned failure");
                std::memset(ptr,0x17,1024);
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                return moe_upload_batch{};
            },
            [&](size_t,const moe_upload_batch &) {}, [&] {++fences;});
    } catch (const std::runtime_error &) {caught=true;}
    CHECK(caught && active==0 && fences==1);
}

static void test_caller_exception_joins_worker() {
    std::vector<uint8_t> stage(8192);
    std::atomic<int> active{0};
    bool caught=false;
    int fences=0;
    try {
        moe_upload_pipeline::execute(stage.data(),stage.size(),3,1024,
            [&](size_t bank,uint8_t * ptr,size_t cap) -> moe_upload_batch {
                CHECK(cap>=1024); ++active;
                struct dec {std::atomic<int>& count; ~dec(){--count;}} guard{active};
                std::memset(ptr,0x3c,1024);
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                if(bank==0) throw std::runtime_error("caller failure");
                return moe_upload_batch{};
            },
            [&](size_t,const moe_upload_batch &) {}, [&] {++fences;});
    } catch (const std::runtime_error &) {caught=true;}
    CHECK(caught && active==0 && fences==0);
}

static void test_submit_on_caller_only() {
    std::vector<uint8_t> stage(16*1024);
    const auto caller=std::this_thread::get_id();
    for (int repeat=0;repeat<40;++repeat) {
        size_t submitted=0;
        bool ran=moe_upload_pipeline::execute(stage.data(),stage.size(),3,128,
            [&](size_t bank,uint8_t * ptr,size_t) -> moe_upload_batch {
                std::memset(ptr,int(bank),128);
                moe_upload_batch result;
                result.spans.push_back({ptr,bank*128,128});
                return result;
            },
            [&](size_t ip,const moe_upload_batch & b) {
                CHECK(std::this_thread::get_id() == caller);
                CHECK(ip==submitted++);
                CHECK(b.spans.size()==1);
            }, [] {CHECK(false);});
        CHECK(ran && submitted==3);
    }
}

// An inline request can fail while preparing a later bank, or after a
// backend accepted an earlier async transfer. Both paths must fence.
static void test_inline_error_lifetime() {
    uint8_t stage[3*512] = {};
    const auto owner = std::this_thread::get_id();
    for (int fault = 0; fault < 3; ++fault) {
        bool pending = false;
        bool caught = false;
        int fences = 0;
        try {
            moe_upload_pipeline::execute(stage, sizeof(stage), 3, 256,
                [&](size_t i, uint8_t * p, size_t n) {
                    CHECK(owner == std::this_thread::get_id());
                    CHECK(p == stage + i*512 && n == 512);
                    if ((fault == 0 && i == 0) || (fault == 1 && i == 1)) {
                        throw std::runtime_error("prepare failed");
                    }
                    return moe_upload_batch{};
                },
                [&](size_t, const moe_upload_batch &) {
                    CHECK(owner == std::this_thread::get_id());
                    pending = true;
                    if (fault == 2) throw std::runtime_error("partial submit");
                },
                [&] { CHECK(pending); pending = false; ++fences; },
                true);
        } catch (const std::runtime_error &) {
            caught = true;
        }
        CHECK(caught && !pending && fences == (fault == 0 ? 0 : 1));
    }
}

int main(){
    test_exact_order_and_disjoint_slices();
    test_no_torn_async_reads();
    test_rejection_without_side_effects();
    test_worker_exception_joins_and_aborts();
    test_caller_exception_joins_worker();
    test_submit_on_caller_only();
    test_inline_error_lifetime();
    std::cout << "P09 host pipeline fixture: 6 PASS; R20 inline exception paths PASS\n";
}
