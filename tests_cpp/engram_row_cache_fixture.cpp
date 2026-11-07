#include "../modules/engram/row_cache.hpp"
#include "../modules/engram/mapped_rows.hpp"
#include <atomic>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>
using llama_exp::engram_row_cache;

static void check(bool v, const char * reason) {
    if (!v) throw std::runtime_error(reason);
}
static constexpr size_t N = 8192;
static constexpr size_t W = 256;
static uint8_t correct(engram_row_cache::row_id row, size_t b) {
    return uint8_t((row*31 + b*17 + b*b) & 255);
}
static std::vector<uint8_t> reference() {
    std::vector<uint8_t> x(N*W);
    for (size_t i = 0; i < N; ++i) for (size_t j = 0; j < W; ++j) x[i*W+j] = correct(i,j);
    return x;
}
static void check_row(const uint8_t * data, size_t row) {
    for (size_t i = 0; i < W; ++i) check(data[i] == correct(row,i), "incorrect quantized row bytes");
}
static void geometry_and_sync() {
    const auto source = reference();
    auto reader = [&](uint64_t id, uint8_t * dst, size_t n) {
        check(n == W && id < N, "bad reader invocation");
        std::memcpy(dst, source.data() + id*W, W);
        return true;
    };
    for (const auto budget : {size_t(0),size_t(W-1),size_t(65)*1024*1024}) {
        bool rejected = false;
        try { engram_row_cache cache(reader,N,W,budget,0); } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "budget should be rejected");
    }
    engram_row_cache cache(reader,N,W,8*W,0);
    std::array<uint8_t,W> dest{};
    check(cache.get(3,dest.data(),W) == engram_row_cache::read_result::source_read, "first miss");
    check_row(dest.data(),3);
    check(cache.get(3,dest.data(),W) == engram_row_cache::read_result::cache_hit, "cached row hit");
    check(cache.get(3,nullptr,W) == engram_row_cache::read_result::invalid_output, "null destination accepted");
    check(cache.get(N,dest.data(),W) == engram_row_cache::read_result::invalid_row, "bad row accepted");
    check(cache.get(2,dest.data(),W-1) == engram_row_cache::read_result::invalid_output, "short destination accepted");
    for (uint64_t i = 0; i < 512; ++i) {
        check(cache.get(i,dest.data(),W) != engram_row_cache::read_result::io_error, "sync I/O error");
        check_row(dest.data(),i);
        check(cache.snapshot().resident_data_bytes <= 8*W, "cache budget overflow");
    }
    check(cache.snapshot().evicted > 0, "no evictions");
    cache.close(); cache.close();
    check(cache.snapshot().resident_rows == 0 && cache.snapshot().stopped, "close did not free payload");
    check(cache.get(3,dest.data(),W) == engram_row_cache::read_result::closed, "read after close");
    check(cache.prefetch(nullptr,0) == 0, "unexpected queue fill");
}
static void async_and_eviction() {
    const auto source = reference();
    std::atomic<size_t> reads{0};
    auto reader = [&](uint64_t id, uint8_t * dst, size_t n) {
        check(n == W && id < N, "bad reader input");
        std::memcpy(dst, source.data()+id*W, W);
        ++reads;
        return true;
    };
    engram_row_cache cache(reader,N,W,16*W,64);
    for (uint64_t round = 0; round < 36; ++round) {
        std::vector<uint64_t> ids;
        for (uint64_t i=0;i<12;++i) ids.push_back((round*73+i*3)%N);
        cache.prefetch(ids.data(),ids.size());
        std::array<uint8_t,W> buf{};
        for (auto id : ids) {
            const auto status = cache.get(id,buf.data(),W,std::chrono::milliseconds(10));
            check(status == engram_row_cache::read_result::cache_hit ||
                  status == engram_row_cache::read_result::source_read, "row unavailable");
            check_row(buf.data(),id);
        }
        check(cache.snapshot().resident_data_bytes<=16*W, "async cache byte limit exceeded");
    }
    const auto before = cache.snapshot();
    check(before.background_reads>0, "worker never executed");
    check(before.synchronous_reads + before.hit > 0, "foreground path unused");
    cache.close();
    const auto after = cache.snapshot();
    check(after.resident_rows == 0 && after.queued_or_inflight == 0, "close leaked rows");
    check(reads.load() == after.background_reads+after.synchronous_reads,
          "unsynchronized callback read counters");
}
static void failure_isolation() {
    const auto source = reference();
    std::atomic<int> fail{1};
    auto reader = [&](uint64_t id, uint8_t * dst, size_t) {
        if (id == 37 && fail.exchange(0) == 1) throw std::runtime_error("fake EIO");
        std::memcpy(dst,source.data()+id*W,W);
        return true;
    };
    engram_row_cache cache(reader,N,W,W*4,0);
    std::array<uint8_t,W> dest{};
    check(cache.get(37,dest.data(),W) == engram_row_cache::read_result::io_error, "EIO not propagated");
    check(cache.snapshot().resident_rows == 0, "partial row cached");
    check(cache.get(37,dest.data(),W) == engram_row_cache::read_result::source_read, "retry failed");
    check_row(dest.data(),37);
    cache.close();
}
static void close_during_blocked_io() {
    std::mutex m;
    std::condition_variable cv;
    bool started = false, release = false;
    std::atomic<int> calls{0};
    auto reader = [&](uint64_t id, uint8_t * dst, size_t len) {
        check(len == W, "wrong byte count");
        ++calls;
        {
            std::unique_lock<std::mutex> lock(m);
            started = true;
            cv.notify_all();
            cv.wait(lock,[&] { return release; });
        }
        for(size_t i=0;i<len;++i) dst[i] = correct(id,i);
        return true;
    };
    engram_row_cache cache(reader,N,W,4*W,4);
    std::array<uint64_t,5> ids{3,3,4,5,6};
    check(cache.prefetch(ids.data(),ids.size()) == 4, "deduplication/queue bound failed");
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return started; });
    }
    // These readers can wait on the prefetch CV while the source callback is
    // blocked. close() must wake them and make each return closed.
    constexpr size_t waiter_count = 8;
    std::atomic<size_t> entering{0}, cancelled{0};
    std::vector<std::thread> waiters;
    for (size_t i = 0; i < waiter_count; ++i) {
        waiters.emplace_back([&] {
            std::array<uint8_t, W> dest{};
            ++entering;
            if (cache.get(ids[0], dest.data(), W, std::chrono::seconds(30)) ==
                    engram_row_cache::read_result::closed) ++cancelled;
        });
    }
    while (entering.load() != waiter_count) std::this_thread::yield();
    std::atomic<bool> closed{false};
    std::thread closer([&] { cache.close(); closed.store(true); });
    // Wait until the close thread has published stopping, then unblock the
    // already running callback. The queued IDs must not start new reads.
    bool saw_stopping = false;
    for (int spin=0; spin<50000; ++spin) {
        if (cache.snapshot().stopped) { saw_stopping = true; break; }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    // The callback is ALWAYS released before failing a test, otherwise stack
    // unwinding would hang while close() correctly waits for a blocked reader.
    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();
    closer.join();
    for (auto & waiter : waiters) waiter.join();
    check(cancelled.load() == waiter_count, "close failed to wake waiting reads");
    check(saw_stopping, "close did not begin");
    check(closed.load() && calls.load() == 1, "close failed to drain exactly one inflight callback");
    check(cache.snapshot().resident_rows == 0, "cache retained bytes after close");
}
static void mapped_strided_rows() {
    // Q6_K payload may be smaller than mmap's configured row stride; preserve
    // the exact payload, never read padding into the cache.
    constexpr size_t ROWS=64, PAYLOAD=210, STRIDE=256;
    std::vector<uint8_t> mapping(ROWS*STRIDE,0xCC);
    for(size_t row=0;row<ROWS;++row) {
        for(size_t b=0;b<PAYLOAD;++b) mapping[row*STRIDE+b]=uint8_t((row*37+b*13)&255);
    }
    llama_exp::mapped_rows reader(mapping.data(),mapping.size(),STRIDE,PAYLOAD);
    engram_row_cache cache(reader,ROWS,PAYLOAD,7*PAYLOAD,16);
    std::array<uint8_t,PAYLOAD> dst{};
    for(size_t row=0;row<ROWS;++row) {
        auto result=cache.get(row,dst.data(),dst.size());
        check(result==engram_row_cache::read_result::cache_hit ||
              result==engram_row_cache::read_result::source_read,"strided row unavailable");
        for(size_t b=0;b<PAYLOAD;++b) check(dst[b]==uint8_t((row*37+b*13)&255),"strided payload mismatch");
    }
    check(!reader(ROWS,dst.data(),PAYLOAD),"out-of-mapping read accepted");
    check(!reader(UINT64_MAX,dst.data(),PAYLOAD),"overflowed row accepted");
    check(!reader(1,dst.data(),PAYLOAD-1),"incorrect byte width accepted");
    cache.close();
    bool bad=false;
    try { llama_exp::mapped_rows invalid(mapping.data(),mapping.size(),16,210); }
    catch(const std::invalid_argument &) { bad=true; }
    check(bad,"invalid GGUF stride accepted");
}
static void multithreaded_gets() {
    const auto source=reference();
    std::atomic<int> inside{0}, simultaneous{0};
    auto reader=[&](uint64_t id,uint8_t * out,size_t n) {
        if (++inside>1) ++simultaneous;
        std::memcpy(out,source.data()+id*W,n);
        --inside;
        return true;
    };
    engram_row_cache cache(reader,N,W,24*W,128);
    std::vector<std::thread> workers;
    for(int thread=0;thread<6;++thread) workers.emplace_back([&,thread] {
        std::mt19937 rng(31337u+thread);
        std::array<uint8_t,W> dest{};
        for(int i=0;i<1200;++i) {
            const uint64_t id=uint64_t(rng()%256);
            const auto status=cache.get(id,dest.data(),W);
            check(status == engram_row_cache::read_result::cache_hit ||
                  status == engram_row_cache::read_result::source_read, "foreground failure");
            check_row(dest.data(),id);
        }
    });
    for(auto & t:workers)t.join();
    check(simultaneous.load()==0,"reader callback invoked concurrently");
    check(cache.snapshot().resident_data_bytes<=24*W,"concurrent cache overflow");
    cache.close();
}
int main() {
    try {
        geometry_and_sync();
        async_and_eviction();
        failure_isolation();
        close_during_blocked_io();
        multithreaded_gets();
        mapped_strided_rows();
        std::cout << "{\"status\":\"PASS\",\"tests\":6,\"gpu_executed\":false,\"model_executed\":false}\n";
        return 0;
    } catch(const std::exception & exc) {
        std::cerr<<"FAIL: "<<exc.what()<<'\n';
        return 1;
    }
}
