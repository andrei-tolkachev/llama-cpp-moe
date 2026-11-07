#include "../modules/moe/upload_completion.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct fake_backend {
    struct copy { uint8_t * dst; const uint8_t * src; size_t count; };
    std::vector<copy> pending;
    size_t completed = 0;
    size_t records = 0;
    size_t event_waits = 0;
    size_t backend_waits = 0;
    size_t destroyed = 0;
    bool events = true;
    bool fail_record = false;
    bool fail_submit = false;
    void enqueue(uint8_t * dst, const uint8_t * src, size_t count) {
        if (fail_submit) throw std::runtime_error("backend enqueue failure");
        pending.push_back({dst,src,count});
    }
    void flush_to(size_t last) {
        assert(last <= pending.size());
        for (; completed < last; ++completed) {
            const auto & c = pending.at(completed);
            std::memcpy(c.dst,c.src,c.count);
        }
    }
};
struct fake_event { fake_backend * backend; size_t marker = 0; bool recorded = false; };
struct adapter {
    using backend_type = fake_backend *;
    using event_type = fake_event *;
    static event_type create(backend_type b) { return b->events ? new fake_event{b,0,false} : nullptr; }
    static void record(event_type e, backend_type b) {
        assert(e && e->backend == b);
        if (b->fail_record) throw std::runtime_error("event record failure");
        e->marker = b->pending.size();
        e->recorded = true;
        ++b->records;
    }
    static void synchronize_event(event_type e) {
        assert(e && e->recorded);
        e->backend->flush_to(e->marker);
        ++e->backend->event_waits;
    }
    static void synchronize_backend(backend_type b) {
        b->flush_to(b->pending.size());
        ++b->backend_waits;
    }
    static void destroy(event_type e) {
        assert(e);
        ++e->backend->destroyed;
        delete e;
    }
};
using fence = llama_exp::moe_upload_completion<adapter>;
using abort_guard = llama_exp::moe_upload_abort_scope<fence>;

void normal_reuse() {
    fake_backend b;
    uint8_t src[8] = {1,2,3,4,5,6,7,8};
    uint8_t dst[8] = {};
    fence f(&b);
    assert(f.event_supported());
    for (int k=0; k<64; ++k) {
        f.begin_submission();
        abort_guard guard(&f);
        b.enqueue(dst,src,8);
        f.record_after_slot_map();
        guard.commit();
        // No added host-side synchronize on the ordinary success path.
        assert(b.completed == size_t(k));
        b.flush_to(b.pending.size()); // the original routing-ID readback sync
        assert(f.retired_after_backend_sync());
    }
    assert(std::memcmp(src,dst,8) == 0);
    assert(b.records == 64 && b.event_waits == 0 && b.backend_waits == 0);
    f.close(); f.close();
    assert(b.destroyed == 1 && f.closed());
}

void graceful_close_pending() {
    fake_backend b;
    uint8_t src[3] = {11,22,33};
    uint8_t dst[3] = {};
    fence f(&b);
    f.begin_submission();
    b.enqueue(dst,src,3);
    f.record_after_slot_map();
    assert(dst[0] == 0);
    f.close();
    assert(std::memcmp(src,dst,3) == 0);
    assert(b.event_waits == 1 && b.backend_waits == 0 && b.destroyed == 1);
}

void eventless_fallback() {
    fake_backend b; b.events = false;
    uint8_t src = 19, dst = 0;
    fence f(&b);
    assert(!f.event_supported());
    f.begin_submission();
    b.enqueue(&dst,&src,1);
    f.record_after_slot_map();
    f.close();
    assert(dst == 19 && b.backend_waits == 1 && b.event_waits == 0 && b.destroyed == 0);
}

void failure_after_enqueue() {
    fake_backend b;
    uint8_t src = 42, dst = 0;
    fence f(&b);
    try {
        f.begin_submission();
        abort_guard guard(&f);
        b.enqueue(&dst,&src,1);
        throw std::runtime_error("failure after one H2D");
    } catch (const std::runtime_error &) {}
    assert(dst == 42 && !f.pending());
    assert(b.backend_waits == 1); // no event was recorded yet
    f.close();
}

void failed_event_record() {
    fake_backend b; b.fail_record = true;
    uint8_t src = 88, dst = 0;
    fence f(&b);
    try {
        f.begin_submission();
        abort_guard guard(&f);
        b.enqueue(&dst,&src,1);
        f.record_after_slot_map();
        assert(false);
    } catch (const std::runtime_error &) {}
    assert(dst == 88 && b.backend_waits == 1 && !f.pending());
    f.close();
}

void rejected_double_begin() {
    fake_backend b;
    fence f(&b);
    f.begin_submission();
    bool threw=false;
    try { f.begin_submission(); } catch (const std::logic_error &) {threw=true;}
    assert(threw);
    f.drain();
    f.close();
    threw=false;
    try {f.begin_submission();} catch(const std::logic_error &){threw=true;}
    assert(threw);
}

void independent_backends() {
    fake_backend a,b; b.events=false;
    uint8_t src_a=7,src_b=9,dst_a=0,dst_b=0;
    fence fa(&a), fb(&b);
    fa.begin_submission(); fb.begin_submission();
    a.enqueue(&dst_a,&src_a,1); b.enqueue(&dst_b,&src_b,1);
    fa.record_after_slot_map();fb.record_after_slot_map();
    fa.close();
    assert(dst_a==7 && dst_b==0 && b.backend_waits==0);
    fb.close();
    assert(dst_b==9 && a.destroyed==1 && b.destroyed==0);
}

void no_false_retirement() {
    fake_backend b;
    fence f(&b);
    assert(!f.retired_after_backend_sync());
    f.begin_submission();
    assert(f.retired_after_backend_sync());
    assert(!f.pending());
    f.close();
}
}
int main(int argc,char **argv) {
    if(argc>1 && std::string(argv[1])=="--unclosed") {
        fake_backend b;
        fence bad(&b);
        return 0; // destructor MUST terminate: unclosed backend ownership
    }
    assert(fence::enabled() == (std::getenv("LLAMA_EXPERIMENTAL_MOE_EVENT_FENCE") &&
          std::string(std::getenv("LLAMA_EXPERIMENTAL_MOE_EVENT_FENCE"))=="1"));
    normal_reuse();
    graceful_close_pending();
    eventless_fallback();
    failure_after_enqueue();
    failed_event_record();
    rejected_double_begin();
    independent_backends();
    no_false_retirement();
    std::cout << "P14B event-completion tests PASS\n";
}
