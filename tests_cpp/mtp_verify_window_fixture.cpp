#include "../modules/speculative/mtp_verify_window.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

struct test_seq {
    std::vector<float> full;
    std::vector<float> retained;
    std::vector<float> pending;
    llama_exp::mtp_verify_window window;
    std::size_t width;

    test_seq(std::size_t n, std::size_t dim, int cap, bool enabled)
          : window(n, cap, enabled), width(dim) {
        full.resize(n * dim);
        for (std::size_t i=0; i<n; ++i) {
            for (std::size_t j=0; j<dim; ++j) {
                // Exact binary float values; no rounding issue in comparisons.
                full[i*dim+j] = float((i*97 + j*7) % 8192);
            }
        }
        retained.assign(full.begin(), full.begin() + window.retained_rows*dim);
        pending.assign(dim, -1.0f);
        if (n) {
            // The full last row remains the pending boundary for the next
            // process() even when the verify prefix is compacted.
            std::memcpy(pending.data(),full.data()+(n-1)*dim,dim*sizeof(float));
        }
    }

    void accept(std::uint16_t a) {
        const auto i = window.accepted_row(a);
        std::memcpy(pending.data(), retained.data()+i*width, width*sizeof(float));
        assert(std::memcmp(pending.data(),full.data()+i*width,width*sizeof(float))==0);
    }
};

int main() {
    const char * env = std::getenv("LLAMA_EXPERIMENTAL_MTP_VERIFY_WINDOW");
    const bool expected = env && std::strcmp(env, "1") == 0;
    assert(llama_exp::mtp_verify_window_enabled() == expected);
    std::size_t cases=0, compacted=0;
    for (std::size_t n : {std::size_t(1),std::size_t(2),std::size_t(3),std::size_t(4),std::size_t(8),std::size_t(9),std::size_t(16),std::size_t(17),std::size_t(33),std::size_t(64),std::size_t(256),std::size_t(1025)}) {
        for (std::size_t dim : {std::size_t(1),std::size_t(3),std::size_t(7),std::size_t(64)}) {
            for (int max : {-1,0,1,2,4,8,16,32}) {
                for (bool enabled : {false,true}) {
                    test_seq seq(n,dim,max,enabled);
                    assert(seq.window.total_rows==n);
                    assert(seq.window.retained_rows>0 && seq.window.retained_rows<=n);
                    const auto bound = std::size_t(std::max(1, max)) + 1;
                    if (enabled && max>=0 && n > bound) {
                        assert(seq.window.retained_rows==bound);
                        ++compacted;
                    } else {
                        assert(seq.window.retained_rows==n);
                    }
                    const auto snapshot=seq.pending;
                    assert(std::memcmp(snapshot.data(),seq.full.data()+(n-1)*dim,dim*sizeof(float))==0);
                    for(std::size_t k=0;k<n && k<45;++k) {
                        if(k >= seq.window.retained_rows) {
                            try {seq.accept(static_cast<std::uint16_t>(k));assert(false);} 
                            catch(const std::out_of_range &) {}
                        } else {
                            seq.accept(static_cast<std::uint16_t>(k));
                        }
                        ++cases;
                    }
                }
            }
        }
    }
    // n_max=0 still permits the first drafted token on this upstream.
    test_seq zero_cap(2, 4, 0, true);
    zero_cap.accept(1);
    assert(zero_cap.pending == std::vector<float>(zero_cap.full.begin()+4, zero_cap.full.end()));

    // Catch-up, acceptance, and last-row carryover for two client sequences.
    struct token { int32_t seq_id; };
    llama_exp::mtp_batch_rows order;
    std::vector<int32_t> first(3), last(3);
    assert(order.index(std::vector<token>{{0},{2},{2},{0},{2},{0}}, first, last));
    assert(first == std::vector<int32_t>({0,-1,1}));
    assert(last == std::vector<int32_t>({5,-1,4}));
    assert(order.count(0,first[0],last[0]) == 3);
    assert(order.count(2,first[2],last[2]) == 3);
    for (int32_t i=0; i<3; ++i) {
        const auto row0 = order.at(0,first[0],i);
        const auto row2 = order.at(2,first[2],i);
        assert(row0 == std::vector<int32_t>({0,3,5})[i]);
        assert(row2 == std::vector<int32_t>({1,2,4})[i]);
        if (i > 0) {
            assert(order.previous(row0) == order.at(0,first[0],i-1));
            assert(order.previous(row2) == order.at(2,first[2],i-1));
        }
    }
    assert(order.index(std::vector<token>{{2},{2},{0}}, first, last));
    assert(order.count(2,first[2],last[2]) == 2);
    assert(order.at(2,first[2],1) == 1 && order.previous(1) == 0);
    assert(!order.index(std::vector<token>{{-1}}, first, last));
    assert(!order.index(std::vector<token>{{3}}, first, last));

    // Special case: acceptance saturates to last row, including >n_rows.
    test_seq total(3,23,2,true);
    total.accept(std::numeric_limits<std::uint16_t>::max());
    bool threw=false;
    try {llama_exp::mtp_verify_window(0,4,true).accepted_row(0);} catch(const std::out_of_range &) {threw=true;}
    assert(threw);
    threw=false;
    try {test_seq insufficient(200,3,4,true);insufficient.accept(9);}catch(const std::out_of_range &) {threw=true;}
    assert(threw);
    // Separate streams remain isolated; no rows cross sequence boundaries.
    test_seq a(140,13,6,true),b(100,13,8,true);
    a.accept(6);b.accept(8);
    assert(std::memcmp(a.pending.data(), b.pending.data(), 13*sizeof(float))!=0);
    std::cout << "P16A mtp verify window: " << cases << " acceptance cases, "
              << compacted << " compact snapshots, parity OK\n";
}
