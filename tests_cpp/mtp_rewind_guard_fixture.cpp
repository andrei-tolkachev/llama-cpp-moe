#include "../modules/speculative/mtp_rewind_guard.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

static bool equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) == 0;
}

int main() {
    const char * env = std::getenv("LLAMA_EXPERIMENTAL_MTP_REWIND_GUARD");
    const bool expected = env != nullptr && std::string(env) == "1";
    assert(llama_exp::mtp_rewind_guard_enabled() == expected);
    llama_exp::mtp_rewind_guard guards[4];
    int checks = 0;
    for (int width : {1,3,32,256,4096}) {
        for (int n_rows : {1,2,3,4,17,128,2048}) {
            for (int accepted=0; accepted<n_rows && accepted<=256; ++accepted) {
                for (int seq=0; seq<4; ++seq) {
                    std::vector<float> original(width), live(width);
                    for (int i=0;i<width;++i) original[i] = float(i + width*seq) * 0.125f;
                    live = original;
                    auto & g=guards[seq];
                    g.checkpoint(live);
                    assert(g.has_checkpoint() && g.checkpoint_bytes() == original.size()*sizeof(float));
                    g.verified(n_rows);
                    g.validate_accept(accepted);
                    for (auto & v:live) v += 41.0f;
                    // A checkpoint restoration must preserve the byte-exact boundary.
                    g.rewind(live);
                    assert(equal(live,original) && !g.can_accept());
                    bool blocked=false;
                    try {g.validate_accept(accepted);} catch(const std::logic_error &){blocked=true;}
                    assert(blocked);
                    g.verified(n_rows); // process(replay) now has fresh target rows
                    g.commit_accept(accepted);
                    assert(!g.has_checkpoint() && !g.can_accept());
                    blocked=false;
                    try {g.commit_accept(0);} catch(const std::logic_error &){blocked=true;}
                    assert(blocked);
                    ++checks;
                }
            }
            // All out-of-window acceptance is rejected before pending_h is touched.
            guards[0].checkpoint(std::vector<float>(width,0.0f));
            guards[0].verified(n_rows);
            bool rejected=false;
            try {guards[0].validate_accept(n_rows);} catch(const std::out_of_range &){rejected=true;}
            assert(rejected);
            guards[0].discard();
            ++checks;
        }
    }
    // Abort and missing snapshot cannot silently recover from an unrelated batch.
    guards[0].checkpoint(std::vector<float>(32,1.0f));
    guards[0].verified(3);
    guards[0].discard();
    bool blocked=false;
    std::vector<float> live(32,2.0f);
    try {guards[0].rewind(live);}catch(const std::logic_error &){blocked=true;}
    assert(blocked);
    assert(std::all_of(live.begin(),live.end(),[](float v){return v==2.0f;}));
    guards[1].checkpoint(std::vector<float>(32,1.0f));
    live.resize(33);
    blocked=false;
    try {guards[1].rewind(live);}catch(const std::logic_error &){blocked=true;}
    assert(blocked);
    guards[1].discard();
    assert(checks>8000);
    std::printf("P16B PASS %d accepted/replay/boundary cases\n", checks);
}
