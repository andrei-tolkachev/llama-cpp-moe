// P15A: compile upstream and alternative CUDA kernel bodies as host functions.
// This tests indexing/arithmetic/predicate, NOT NVCC, GPU scheduling or speed.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

struct fake_dim3 { int64_t x = 0; };
static fake_dim3 blockIdx, threadIdx, blockDim;
#define __global__
#define ggml_cuda_pdl_lc() ((void) 0)
#define ggml_cuda_pdl_sync() ((void) 0)
#include "hc_post_reference_original.hpp"
#include "../modules/cuda/hc_post_reuse.cuh"

struct layout {
    int64_t ne, nt, sx0, sx1, sr0, sr1, sr2, sp0, sp1, sc0, sc1, sc2, sd0, sd1, sd2;
};

static void old_kernel(const layout & l, const float * x, const float * r, const float * p,
        const float * c, float * dst) {
    blockDim.x = 256;
    const int64_t nr = l.ne*4*l.nt;
    for (int64_t i=0; i<nr + 256; ++i) {
        blockIdx.x = i / 256;
        threadIdx.x = i % 256;
        dsv4_hc_post_f32<true, false>(x, r, p, c, dst,
                l.ne, 4, l.nt, l.sx0, l.sx1, l.sr0, l.sr1, l.sr2,
                l.sp0, l.sp1, l.sc0, l.sc1, l.sc2,
                l.sd0, l.sd1, l.sd2, 1.0f, 1.0f);
    }
}

static void new_kernel(const layout & l, const float * x, const float * r, const float * p,
        const float * c, float * dst) {
    blockDim.x = 256;
    const int64_t nr = l.ne*l.nt;
    for (int64_t i=0; i<nr + 256; ++i) {
        blockIdx.x = i / 256;
        threadIdx.x = i % 256;
        dsv4_hc_post_reuse_f32(x, r, p, c, dst,
                l.ne, l.nt, l.sx0, l.sx1, l.sr0, l.sr1, l.sr2,
                l.sp0, l.sp1, l.sc0, l.sc1, l.sc2,
                l.sd0, l.sd1, l.sd2);
    }
}

static void fill(std::vector<float> & v, std::mt19937 & rng) {
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    for (float & f : v) f = distribution(rng);
}

static void check_case(int64_t ne, int64_t nt, bool strided, std::mt19937 & rng) {
    const int64_t a = strided ? 2 : 1;
    const int64_t b = strided ? 3 : 1;
    layout l{};
    l.ne=ne; l.nt=nt;
    l.sx0=a; l.sx1=ne*a + (strided ? 7 : 0);
    l.sr0=a; l.sr1=ne*a + (strided ? 3 : 0); l.sr2=4*l.sr1 + (strided ? 11 : 0);
    l.sp0=b; l.sp1=4*b + (strided ? 7 : 0);
    l.sc0=b; l.sc1=4*b + (strided ? 5 : 0); l.sc2=4*l.sc1 + (strided ? 9 : 0);
    l.sd0=a; l.sd1=ne*a + (strided ? 5 : 0); l.sd2=4*l.sd1 + (strided ? 13 : 0);
    std::vector<float> x((nt-1)*l.sx1 + (ne-1)*l.sx0 + 7);
    std::vector<float> r((nt-1)*l.sr2 + 3*l.sr1 + (ne-1)*l.sr0 + 7);
    std::vector<float> p((nt-1)*l.sp1 + 3*l.sp0 + 7);
    std::vector<float> c((nt-1)*l.sc2 + 3*l.sc1 + 3*l.sc0 + 7);
    const size_t out_size=(nt-1)*l.sd2 + 3*l.sd1 + (ne-1)*l.sd0 + 7;
    std::vector<float> baseline(out_size, 13.5f), candidate(out_size, 13.5f);
    fill(x,rng);fill(r,rng);fill(p,rng);fill(c,rng);
    old_kernel(l,x.data(),r.data(),p.data(),c.data(),baseline.data());
    new_kernel(l,x.data(),r.data(),p.data(),c.data(),candidate.data());
    if (std::memcmp(baseline.data(),candidate.data(),out_size*sizeof(float)) != 0) {
        for (size_t i=0;i<out_size;++i) if (baseline[i]!=candidate[i]) {
            fprintf(stderr,"P15 parity mismatch (ne=%lld,nt=%lld,stride=%d,index=%zu) legacy=%g candidate=%g\n",
                (long long)ne,(long long)nt,(int)strided,i,baseline[i],candidate[i]);
            std::abort();
        }
        std::abort();
    }
}

int main() {
    const auto admits = [](int64_t ne,int64_t hc,int64_t nt,bool comb,bool gate) {
        return llama_exp_dsv4_hc_post_reuse_eligible(ne,hc,nt,comb,gate);
    };
    assert(admits(1024,4,64,true,false));
    assert(admits(8192,4,8,true,false));
    assert(!admits(512,4,8,true,false));
    assert(!admits(8192,4,1,true,false));
    assert(!admits(1024,3,128,true,false));
    assert(!admits(1024,4,128,false,false));
    assert(!admits(1024,4,128,true,true));
    assert(!admits(0,4,128,true,false));
    assert(!admits(1024,4,-1,true,false));
    assert(!admits(std::numeric_limits<int64_t>::max(),4,8,true,false));
    std::mt19937 rng(0xD5A41u);
    int count=0;
    for (auto ne : {1,15,255,256,512,1024,2048}) {
        for (auto nt : {1,2,8,17,64}) {
            if (ne*nt>65536) continue;
            for (bool strided : {false,true}) {
                check_case(ne,nt,strided,rng);
                count++;
            }
        }
    }
    printf("P15A host arithmetic parity: %d shapes/layouts PASS\n",count);
}
