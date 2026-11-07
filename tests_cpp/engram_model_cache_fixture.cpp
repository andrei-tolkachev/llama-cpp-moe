#include "../modules/engram/graph_stage.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr size_t rows = 2048;
constexpr size_t payload = 210;
constexpr size_t stride = 224;
using llama_exp::engram_graph_stage;
using llama_exp::engram_model_cache;
using llama_exp::mapped_rows;
using llama_exp::engram_row_cache;

void ck(bool v, const char * why) { if (!v) throw std::runtime_error(why); }
uint8_t byte_for(int32_t row, size_t offset) {
    return uint8_t((uint64_t(row)*31 + offset*13 + ((row ^ int(offset))*7)) & 255);
}
std::vector<uint8_t> mapping() {
    std::vector<uint8_t> data(rows * stride, 0xCA);
    for (size_t id = 0; id < rows; ++id) {
        for (size_t b = 0; b < payload; ++b) data[id*stride+b] = byte_for(int32_t(id),b);
    }
    return data;
}
void parity(const engram_graph_stage & stage, const std::vector<int32_t> & ids) {
    ck(stage.bytes() == ids.size()*payload, "wrong raw stage length");
    for (size_t i=0; i<ids.size(); ++i) {
        for(size_t b=0;b<payload;++b) {
            ck(stage.data()[i*payload+b]==byte_for(ids[i],b), "raw byte mismatch");
        }
    }
}
void reuse_between_graphs() {
    auto raw = mapping();
    const auto saved = raw;
    mapped_rows view(raw.data(), raw.size(), stride, payload);
    auto model = std::make_shared<engram_model_cache>(view, rows, 12*payload, 16);
    ck(model->matches(view, rows), "same source rejected");
    ck(!model->matches(view, rows-1), "different row count accepted");
    auto first = std::make_unique<engram_graph_stage>(model);
    auto second = std::make_unique<engram_graph_stage>(model);
    const std::vector<int32_t> ids{12,4,12,2000,9,4,17,19};
    ck(first->stage(ids.data(),ids.size()), "first graph failed");
    parity(*first, ids);
    const auto first_count = model->counters();
    ck(second->stage(ids.data(),ids.size()), "second graph failed");
    parity(*second, ids);
    const auto second_count = model->counters();
    ck(second_count.hit >= first_count.hit + ids.size(), "graphs did not share cache");
    ck(second_count.resident_data_bytes <= 12*payload, "cache exceeded budget");
    ck(raw == saved, "borrowed source mutated");
    first.reset();
    second.reset();
    ck(!model->counters().stopped, "graph destructor incorrectly closed model cache");
    model->close();
    ck(model->counters().stopped && model->counters().resident_rows == 0, "model close did not release payload");
}
void async_and_duplicates() {
    auto raw = mapping();
    auto model = std::make_shared<engram_model_cache>(mapped_rows(raw.data(), raw.size(), stride, payload),
                                                     rows, 64*payload, 32);
    engram_graph_stage stage(model);
    std::vector<int32_t> ids(512);
    for(int i=0;i<512;++i) ids[i] = (i/2)%rows;
    ids[4] = -1;
    ids[6] = INT32_MAX;
    const size_t queued = stage.prefetch(ids.data(),ids.size());
    ck(queued > 0 && queued <= 512, "async queue not bounded");
    std::vector<int32_t> valid{4,5,6,7,8,9,10,11,12};
    ck(stage.stage(valid.data(),valid.size()), "foreground failed under async contention");
    parity(stage,valid);
    const auto snap = stage.counters();
    ck(snap.queued == queued, "prefetch accounting inaccurate");
    ck(snap.resident_data_bytes<=64*payload, "async limit exceeded");
    ck(snap.queued_or_inflight<=33, "async queue overflowed");
    const std::vector<int32_t> invalid{12,-1,13};
    ck(!stage.stage(invalid.data(),invalid.size()), "stage accepted invalid row ID");
    model->close();
    ck(!stage.stage(valid.data(), valid.size()), "closed cache served stale model rows");
    ck(stage.prefetch(valid.data(),valid.size())==0, "prefetch allowed after close");
}
void close_while_worker_reads() {
    // Explicit mapping lifetime contract: never unmap while a callback is active.
    auto raw=mapping();
    std::mutex mu;
    std::condition_variable cv;
    bool started=false, release=false;
    std::atomic<int> callbacks{0};
    // The row cache is the I/O owner used by the model cache, with controllable
    // blocked source read for deterministic model-unload drain verification.
    engram_row_cache cache([&](uint64_t id, uint8_t * dest, size_t n) {
        callbacks.fetch_add(1);
        {
            std::unique_lock<std::mutex> lk(mu);
            started=true; cv.notify_all();
            cv.wait(lk,[&] {return release;});
        }
        std::memcpy(dest,raw.data()+id*stride,n);
        return true;
    },rows,payload,4*payload,4);
    const uint64_t targets[3] = {12,13,14};
    ck(cache.prefetch(targets,3)==3, "worker did not accept task");
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk,[&] {return started;});
    }
    std::atomic<bool> closed{false};
    std::thread unload([&] {cache.close();closed=true;});
    bool saw_stop=false;
    for(int t=0;t<2000;++t) {
        if(cache.snapshot().stopped) {saw_stop=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ck(saw_stop, "model close not signaled");
    ck(!closed, "model closed before mapped reader finished");
    {
        std::lock_guard<std::mutex> lk(mu); release=true;
    }
    cv.notify_all(); unload.join();
    ck(closed && callbacks == 1, "unload failed to drain worker");
    raw.clear(); raw.shrink_to_fit(); // only after worker has terminated
    ck(cache.snapshot().stopped && cache.snapshot().resident_data_bytes==0, "cache retained payload after close");
}
void multiple_simultaneous_graphs() {
    auto raw=mapping();
    auto model=std::make_shared<engram_model_cache>(mapped_rows(raw.data(),raw.size(),stride,payload),
                                                    rows,128*payload,48);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for(int t=0;t<6;++t) threads.emplace_back([&,t] {
        engram_graph_stage graph(model);
        std::vector<int32_t> ids(17);
        for(int r=0;r<140;++r) {
            for(int j=0;j<17;++j) ids[j]=int32_t((t*37+r*71+j*13)%rows);
            graph.prefetch(ids.data(),ids.size());
            if(!graph.stage(ids.data(),ids.size())){++failures;continue;}
            for(size_t i=0;i<ids.size();++i) for(size_t b=0;b<payload;++b)
                if(graph.data()[i*payload+b]!=byte_for(ids[i],b)) ++failures;
        }
    });
    for(auto &t:threads)t.join();
    ck(failures==0, "concurrent graph output corrupted");
    ck(model->counters().resident_data_bytes<=128*payload,"concurrent cache payload overflowed");
    model->close();
}
} // namespace
int main() {
    try {
        reuse_between_graphs();
        async_and_duplicates();
        close_while_worker_reads();
        multiple_simultaneous_graphs();
        std::cout << "{\"status\":\"PASS\",\"scenarios\":4,\"gpu\":false,\"model\":false}\n";
        return 0;
    } catch(const std::exception & e) {
        std::cerr << "FAIL " << e.what() << "\n";
        return 1;
    }
}
