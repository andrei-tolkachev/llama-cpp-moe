#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void fill(ggml_tensor * cpu, ggml_tensor * gpu, std::mt19937 & rng) {
    std::vector<float> values(cpu->ne[0]*cpu->ne[1]);
    std::vector<float> importance(cpu->ne[0],1.f);
    std::vector<char> bytes(ggml_nbytes(cpu),0);
    std::normal_distribution<float> normal(0.f,0.04f);
    ggml_quantize_init(cpu->type);
    for (int e=0;e<4;++e) {
        for (auto & x:values) x=normal(rng);
        if (cpu->type==GGML_TYPE_F32) std::memcpy(bytes.data()+e*cpu->nb[2],values.data(),values.size()*sizeof(float));
        else ggml_quantize_chunk(cpu->type,values.data(),bytes.data()+e*cpu->nb[2],0,cpu->ne[1],cpu->ne[0],importance.data());
    }
    ggml_backend_tensor_set(cpu,bytes.data(),0,bytes.size());
    ggml_backend_tensor_set(gpu,bytes.data(),0,bytes.size());
}

static bool check(ggml_backend_t gpu,ggml_backend_t cpu,ggml_type type,bool fused) {
    ggml_init_params params={2*1024*1024,nullptr,true};
    auto * cw=ggml_init(params);auto * gw=ggml_init(params);auto * ctx=ggml_init(params);
    ggml_tensor * weights[2][3];
    for (int side=0;side<2;++side) {
        auto * wc=side?cw:gw;
        weights[side][0]=ggml_new_tensor_3d(wc,type,256,fused?1024:512,5);
        weights[side][1]=fused?nullptr:ggml_new_tensor_3d(wc,type,256,512,5);
        weights[side][2]=ggml_new_tensor_3d(wc,type,512,256,5);
    }
    auto * x=ggml_new_tensor_3d(cw,GGML_TYPE_F32,256,1,1);
    auto * ids_all=ggml_new_tensor_2d(cw,GGML_TYPE_I32,4,1);
    auto * ids_gpu=ggml_new_tensor_2d(cw,GGML_TYPE_I32,4,1);
    auto * ids_cpu=ggml_new_tensor_2d(cw,GGML_TYPE_I32,4,1);
    ggml_set_input(x);ggml_set_input(ids_all);ggml_set_input(ids_gpu);ggml_set_input(ids_cpu);
    auto * cb=ggml_backend_alloc_ctx_tensors(cw,cpu);auto * gb=ggml_backend_alloc_ctx_tensors(gw,gpu);
    if (!cb||!gb) return false;
    ggml_backend_buffer_set_usage(gb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(47+type+101*fused);
    for (int j=0;j<3;++j) if (weights[1][j]) fill(weights[1][j],weights[0][j],rng);
    std::vector<float> input(256);
    for (auto & f:input) f=float(int(rng()%1024)-512)/512.f;
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    int32_t all[4]={1,3,0,2};ggml_backend_tensor_set(ids_all,all,0,sizeof(all));
    ggml_backend_t backends[]={gpu,cpu};
    auto * sched=ggml_backend_sched_new(backends,nullptr,2,512,false,true);
    ggml_backend_sched_reset(sched);
    auto * graph=ggml_new_graph_custom(ctx,512,false);
    auto lane=[&](int side,ggml_tensor * ids) {
        int first=ggml_graph_n_nodes(graph);
        auto * g=ggml_mul_mat_id(ctx,weights[side][0],x,ids);
        ggml_tensor * u;
        if (fused) {
            u=ggml_view_3d(ctx,g,512,4,1,g->nb[1],g->nb[2],512*sizeof(float));
            g=ggml_view_3d(ctx,g,512,4,1,g->nb[1],g->nb[2],0);
        } else u=ggml_mul_mat_id(ctx,weights[side][1],x,ids);
        auto * out=ggml_mul_mat_id(ctx,weights[side][2],ggml_swiglu_split(ctx,g,u),ids);
        ggml_build_forward_expand(graph,out);
        for (int i=first;i<ggml_graph_n_nodes(graph);++i) ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(graph,i),side?cpu:gpu);
        return out;
    };
    auto * ref_gpu=lane(0,ids_all);auto * ref_cpu=lane(1,ids_all);
    auto * split_gpu=lane(0,ids_gpu);auto * split_cpu=lane(1,ids_cpu);
    auto * merged=ggml_add(ctx,split_gpu,split_cpu);
    ggml_backend_sched_set_tensor_backend(sched,merged,gpu);ggml_build_forward_expand(graph,merged);
    ggml_set_output(ref_gpu);ggml_set_output(ref_cpu);ggml_set_output(merged);ggml_set_output(split_gpu);ggml_set_output(split_cpu);
    if (!ggml_backend_sched_alloc_graph(sched,graph)) return false;
    bool ok=true;
    for (int mask=0;mask<16&&ok;++mask) {
        int32_t gi[4],ci[4];
        for (int i=0;i<4;++i) {bool host=mask&(1<<i);gi[i]=host?4:all[i];ci[i]=host?all[i]:-1;}
        ggml_backend_tensor_set(ids_gpu,gi,0,sizeof(gi));ggml_backend_tensor_set(ids_cpu,ci,0,sizeof(ci));
        if (ggml_backend_sched_graph_compute(sched,graph)!=GGML_STATUS_SUCCESS) {ok=false;break;}
        std::vector<float> a(1024),b(1024),c(1024);
        ggml_backend_tensor_get(ref_gpu,a.data(),0,a.size()*4);ggml_backend_tensor_get(ref_cpu,b.data(),0,b.size()*4);ggml_backend_tensor_get(merged,c.data(),0,c.size()*4);
        for (int i=0;i<1024;++i) {
            float ref=(mask&(1<<(i/256)))?b[i]:a[i];
            if (!std::isfinite(c[i])||c[i]!=ref) {
                int32_t dbg[4];ggml_backend_tensor_get(ids_cpu,dbg,0,sizeof(dbg));printf("ids=%d,%d,%d,%d backend=%s\n",dbg[0],dbg[1],dbg[2],dbg[3],ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched,split_cpu)));float sg[1024],sc[1024];ggml_backend_tensor_get(split_gpu,sg,0,sizeof(sg));ggml_backend_tensor_get(split_cpu,sc,0,sizeof(sc));
                printf("FAIL %s fused=%d mask=%d row=%d ref=%a got=%a gpu=%a cpu=%a refcpu=%a\n",ggml_type_name(type),fused,mask,i,ref,c[i],sg[i],sc[i],b[i]);ok=false;break;
            }
        }
    }
    if (ok) printf("PASS %s fused=%d actualCUDA_CPU partitions=16 exact_values=16384\n",ggml_type_name(type),fused);
    ggml_backend_sched_free(sched);ggml_backend_buffer_free(cb);ggml_backend_buffer_free(gb);ggml_free(ctx);ggml_free(cw);ggml_free(gw);
    return ok;
}

int main() {
    auto * cpu=ggml_backend_cpu_init();auto * gpu=ggml_backend_cuda_init(0);
    if (!cpu||!gpu) return 2;
    ggml_backend_cpu_set_n_threads(cpu,4);
    for (auto type:{GGML_TYPE_Q4_K,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_S,GGML_TYPE_MXFP4}) {
        for (bool fused:{false,true}) if (!check(gpu,cpu,type,fused)) return 1;
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);
    puts("PASS actualCUDA_CPU 160partitions 163840values");
}
