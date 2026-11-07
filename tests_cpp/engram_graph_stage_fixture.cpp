#include "../modules/engram/graph_stage.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t row_bytes = 210; // DeepSeek V4.1 Engram Q6_K/256 row bytes
constexpr size_t stride = 224;    // Exercise padding in a mapped GGUF-like region
constexpr uint64_t n_rows = 512;

void require(bool v, const char * msg) { if (!v) throw std::runtime_error(msg); }
uint8_t value(size_t r, size_t k) { return uint8_t((r*19 + k*41 + (k ^ r)) & 0xff); }

std::vector<uint8_t> make_source() {
    std::vector<uint8_t> src(n_rows*stride, 0xcc);
    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t k = 0; k < row_bytes; ++k) src[row*stride+k] = value(row,k);
    }
    return src;
}
void check_stage(const llama_exp::engram_graph_stage & stage,
                 const std::vector<int32_t> & ids) {
    require(stage.bytes() == ids.size()*row_bytes, "staged length mismatch");
    for (size_t i=0; i<ids.size(); ++i)
        for (size_t k=0; k<row_bytes; ++k)
            require(stage.data()[i*row_bytes+k] == value(ids[i],k), "wrong row bytes/order");
}

void exact_raw_stage_and_cache() {
    const auto raw = make_source();
    const auto original = raw;
    llama_exp::mapped_rows view(raw.data(),raw.size(),stride,row_bytes);
    llama_exp::engram_graph_stage stage(view,n_rows,row_bytes*8);
    const std::vector<int32_t> ids{10, 10, 11, 511, 0, 255, 11, 10};
    require(stage.stage(ids.data(),ids.size()),"initial stage failed");
    check_stage(stage,ids);
    require(stage.counters().hit>=3, "repeat rows should hit cache");
    require(stage.counters().resident_data_bytes<=row_bytes*8, "budget exceeded");
    for (int round=0; round<128; ++round) {
        std::vector<int32_t> next;
        for(int j=0;j<40;++j) next.push_back(int32_t((round*73+j*11)%n_rows));
        require(stage.stage(next.data(),next.size()),"repeated stage failed");
        check_stage(stage,next);
        require(stage.counters().resident_data_bytes<=row_bytes*8,"cache budget exceeded");
    }
    require(stage.counters().evicted>0, "missing bounded LRU eviction");
    require(raw==original,"mmap source was changed");
}
void reject_invalid_without_dirtying_previous_buffer() {
    auto raw=make_source();
    llama_exp::engram_graph_stage stage(llama_exp::mapped_rows(raw.data(),raw.size(),stride,row_bytes),n_rows);
    std::vector<int32_t> good{1,2,3};
    require(stage.stage(good.data(),good.size()),"good stage failed");
    const std::vector<uint8_t> saved(stage.data(),stage.data()+stage.bytes());
    const int32_t negative[] = {0,-1,5};
    const int32_t overrun[] = {0,512,5};
    require(!stage.stage(negative,3),"negative index accepted");
    require(!stage.stage(overrun,3),"oversized index accepted");
    require(!stage.stage(nullptr,3),"null pointer accepted");
    require(!stage.stage(good.data(),0),"empty stage accepted");
    require(std::vector<uint8_t>(stage.data(),stage.data()+stage.bytes()) == saved,
            "failed validation changed previous input tensor data");
}
void eligibility_and_overflow() {
    using S=llama_exp::engram_graph_stage;
    require(S::eligible(n_rows,row_bytes,stride,n_rows*stride,64,3),"valid staging geometry refused");
    require(!S::eligible(n_rows,row_bytes,stride,n_rows*stride,0,3),"empty stage accepted");
    require(!S::eligible(n_rows,row_bytes,stride,n_rows*stride,64,0),"empty modules accepted");
    require(!S::eligible(n_rows,row_bytes,stride,row_bytes,64,1),"truncated mmap accepted");
    require(!S::eligible(n_rows,0,stride,n_rows*stride,64,1),"zero bytes accepted");
    require(!S::eligible(n_rows,row_bytes,100,n_rows*stride,64,1),"short stride accepted");
    require(!S::eligible(n_rows,row_bytes,stride,n_rows*stride,uint64_t(INT32_MAX)+1,1),"int32 overflow accepted");
    require(!S::eligible(n_rows,row_bytes,stride,n_rows*stride,200000,10000),"budget product overflow accepted");
    require(!S::eligible(std::numeric_limits<uint64_t>::max(),row_bytes,stride,n_rows*stride,1,1),
            "mmap range overflow accepted");
}
void random_row_byte_parity() {
    auto raw=make_source();
    llama_exp::engram_graph_stage stage(llama_exp::mapped_rows(raw.data(),raw.size(),stride,row_bytes),n_rows);
    std::mt19937 rng(0xC0FFEE);
    for (int trial=0;trial<180;++trial) {
        std::vector<int32_t> ids;
        for (int i=0;i<1+trial%160;++i) ids.push_back(int32_t(rng()%n_rows));
        require(stage.stage(ids.data(),ids.size()),"random stage failure");
        check_stage(stage,ids);
    }
}
}

int main() {
    try {
        exact_raw_stage_and_cache();
        reject_invalid_without_dirtying_previous_buffer();
        eligibility_and_overflow();
        random_row_byte_parity();
        std::cout << "{\"status\":\"PASS\",\"scenarios\":4,\"gpu\":false,\"model\":false}\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL " << e.what() << "\n";
        return 1;
    }
}
