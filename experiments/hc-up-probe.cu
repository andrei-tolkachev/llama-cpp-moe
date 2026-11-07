// Prototype: apply Strata's projection/mixing fusion at the GGML Q8 dot boundary.
// The baseline below is the unmodified accepted GGML CUDA implementation.
#include "mmvq.cu"
#include "dsv4-hc.cu"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

__global__ void hc_up_q8_fused(const void * weights, const block_q8_1 * act,
        const float * x, float * out, int k, int embd, float scale) {
    constexpr int warp = 32, warps = 4, rows = 4;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int tid = warp * threadIdx.y + threadIdx.x;
    const int blocks = k / QK8_0;
    const int e = blockIdx.x;
    float tmp[rows] = {};
    ggml_cuda_pdl_sync();
    for (int b = tid / (qi/vdr); b < blocks; b += vdr * warps * warp / qi) {
        const int qs = vdr * (tid % (qi/vdr));
        #pragma unroll
        for (int h = 0; h < rows; ++h) {
            tmp[h] += vec_dot_q8_0_q8_1(weights, act+b, (e+h*embd)*blocks+b, qs);
        }
    }
    __shared__ float partial[warps-1][rows][warp];
    if (threadIdx.y > 0) {
        #pragma unroll
        for (int h = 0; h < rows; ++h) partial[threadIdx.y-1][h][threadIdx.x]=tmp[h];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    float sum=0.0f;
    #pragma unroll
    for (int h = 0; h < rows; ++h) {
        #pragma unroll
        for (int w = 0; w < warps-1; ++w) tmp[h] += partial[w][h][threadIdx.x];
        tmp[h]=warp_reduce_sum<warp>(tmp[h]);
        if (threadIdx.x==0) {
            const float gate=1.0f/(1.0f+expf(-tmp[h]));
            sum += x[e+h*embd]*gate;
        }
    }
    if (threadIdx.x==0) out[e]=scale*sum;
}

template<class F> cudaGraphExec_t capture(cudaStream_t st,F f) {
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<32;++i) f();
    CUDA_CHECK(cudaStreamEndCapture(st,&graph));
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec,graph,0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return exec;
}
float measure(cudaGraphExec_t graph,cudaStream_t st) {
    for(int i=0;i<5;++i) CUDA_CHECK(cudaGraphLaunch(graph,st));
    cudaEvent_t a,b;
    CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a,st));
    for(int i=0;i<100;++i) CUDA_CHECK(cudaGraphLaunch(graph,st));
    CUDA_CHECK(cudaEventRecord(b,st));CUDA_CHECK(cudaEventSynchronize(b));
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
    CUDA_CHECK(cudaEventDestroy(a));CUDA_CHECK(cudaEventDestroy(b));
    return ms*1000/(100*32);
}
int main() {
    ggml_backend_t backend=ggml_backend_cuda_init(0);
    if(!backend) return 2;
    const auto & dev=ggml_cuda_info().devices[0];
    if(get_device_table_id(dev.cc)!=MMVQ_PARAMETERS_GENERIC || dev.warp_size!=32) return 3;
    cudaStream_t st;CUDA_CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    size_t total=0,mismatch=0;float maxerr=0;int cases=0;
    uint32_t rng=18791;
    auto randf=[&](){rng=rng*1664525u+1013904223u;return (int(rng>>8)-8388608)/8388608.0f;};
    for(int k: {64,128,256,320,512,768}) for(int embd:{64,2560}) {
        int pad=GGML_PAD(k,MATRIX_ROW_PADDING);
        size_t nw=size_t(k/32)*embd*4;
        std::vector<block_q8_0> hw(nw);
        for(auto & b:hw) {b.d=__float2half(0.002f+std::fabs(randf())*0.04f);for(auto & q:b.qs)q=int8_t(randf()*127);}
        std::vector<float> ha(k),hx(embd*4),ref(embd),got(embd);
        void * w;float * a,* x,* gate,* y0,* y1;block_q8_1 * aq;
        CUDA_CHECK(cudaMalloc(&w,nw*sizeof(block_q8_0)));CUDA_CHECK(cudaMalloc(&a,k*4));
        CUDA_CHECK(cudaMalloc(&x,embd*4*4));CUDA_CHECK(cudaMalloc(&gate,embd*4*4));
        CUDA_CHECK(cudaMalloc(&y0,embd*4));CUDA_CHECK(cudaMalloc(&y1,embd*4));
        CUDA_CHECK(cudaMalloc(&aq,pad/32*sizeof(block_q8_1)));
        CUDA_CHECK(cudaMemcpyAsync(w,hw.data(),nw*sizeof(block_q8_0),cudaMemcpyHostToDevice,st));
        auto quant=[&](){quantize_row_q8_1_cuda(a,nullptr,aq,GGML_TYPE_Q8_0,k,k,k,k,pad,1,1,1,st);};
        auto original=[&](){
            quant();
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>(w,aq,nullptr,{},gate,k,embd*4,1,
                k/32,pad/32,embd*4,1,1,1,k/32*embd*4,pad/32,embd*4,
                1,1,k/32*embd*4,pad/32,embd*4,0,st);
            ggml_cuda_kernel_launch_params lp(dim3((embd+255)/256),dim3(256),0,st);
            ggml_cuda_kernel_launch(dsv4_hc_pre_f32<true>,lp,x,gate,y0,int64_t(embd),int64_t(4),int64_t(1),
                int64_t(1),int64_t(embd),int64_t(embd*4),int64_t(1),int64_t(embd),int64_t(embd*4),int64_t(1),int64_t(embd),0.25f);
        };
        auto fused=[&](){quant();ggml_cuda_kernel_launch_params lp(dim3(embd),dim3(32,4),0,st);
            ggml_cuda_kernel_launch(hc_up_q8_fused,lp,w,aq,x,y1,k,embd,0.25f);};
        for(float mag:{0.0f,0.0001f,0.1f,1.0f,100.0f}) {
            for(auto & v:ha)v=randf()*mag;for(auto & v:hx)v=randf()*4;
            CUDA_CHECK(cudaMemcpyAsync(a,ha.data(),k*4,cudaMemcpyHostToDevice,st));
            CUDA_CHECK(cudaMemcpyAsync(x,hx.data(),embd*4*4,cudaMemcpyHostToDevice,st));
            original();fused();CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaMemcpy(ref.data(),y0,embd*4,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(got.data(),y1,embd*4,cudaMemcpyDeviceToHost));
            for(int i=0;i<embd;++i){++total;if(std::memcmp(&ref[i],&got[i],4))++mismatch;
                if(!std::isfinite(ref[i])||!std::isfinite(got[i])) return 4;
                maxerr=std::max(maxerr,std::fabs(ref[i]-got[i]));}
            ++cases;
        }
        for(auto & v:ha)v=randf()*0.1f;
        CUDA_CHECK(cudaMemcpyAsync(a,ha.data(),k*4,cudaMemcpyHostToDevice,st));CUDA_CHECK(cudaStreamSynchronize(st));
        auto g0=capture(st,original),g1=capture(st,fused);
        float a0=measure(g0,st),b0=measure(g1,st),b1=measure(g1,st),a1=measure(g0,st);
        std::printf("BENCH k=%d embd=%d baseline_us=%.6f fused_us=%.6f speedup=%.5f\n",k,embd,(a0+a1)/2,(b0+b1)/2,(a0+a1)/(b0+b1));std::fflush(stdout);
        CUDA_CHECK(cudaGraphExecDestroy(g0));CUDA_CHECK(cudaGraphExecDestroy(g1));
        for(void * p:{w,(void*)a,(void*)x,(void*)gate,(void*)y0,(void*)y1,(void*)aq})CUDA_CHECK(cudaFree(p));
    }
    std::printf("CHECK cases=%d floats=%zu mismatches=%zu max_abs_error=%.9g\n",cases,total,mismatch,maxerr);
    CUDA_CHECK(cudaStreamDestroy(st));ggml_backend_free(backend);
    return mismatch?1:0;
}
