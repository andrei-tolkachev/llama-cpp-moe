#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "src/llama-experimental/cpu_gate_up.hpp"
#include <random>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
static void fill(ggml_tensor * t,std::mt19937 & r) {
 std::vector<float> f(t->ne[0]*t->ne[1]),im(t->ne[0],1.f);std::vector<char> b(ggml_nbytes(t));std::normal_distribution<float> n(0.f,.04f);
 ggml_quantize_init(t->type);
 for(int e=0;e<t->ne[2];++e){for(auto & x:f)x=n(r);ggml_quantize_chunk(t->type,f.data(),b.data()+e*t->nb[2],0,t->ne[1],t->ne[0],im.data());}
 ggml_backend_tensor_set(t,b.data(),0,b.size());
}
static bool test(ggml_backend_t cpu,ggml_type gtype,ggml_type utype,ggml_type dtype,int hidden,int width,int threads,bool bench) {
 ggml_backend_cpu_set_n_threads(cpu,threads);ggml_init_params ip={4*1024*1024,nullptr,true};auto * w=ggml_init(ip);auto * ctx=ggml_init(ip);
 auto * g=ggml_new_tensor_3d(w,gtype,width,hidden,4);auto * u=ggml_new_tensor_3d(w,utype,width,hidden,4);auto * d=ggml_new_tensor_3d(w,dtype,hidden,width,4);
 auto * x=ggml_new_tensor_3d(w,GGML_TYPE_F32,width,1,1);auto * ids=ggml_new_tensor_2d(w,GGML_TYPE_I32,4,1);ggml_set_input(x);ggml_set_input(ids);
 auto * wb=ggml_backend_alloc_ctx_tensors(w,cpu);if(!wb)return false;std::mt19937 rng(734+gtype+utype);fill(g,rng);fill(u,rng);fill(d,rng);
 std::vector<float> input(width);for(auto & f:input)f=float(int(rng()%2048)-1024)/64.f;ggml_backend_tensor_set(x,input.data(),0,width*4);
 float limit=10.f;auto * rg=ggml_mul_mat_id(ctx,g,x,ids);auto * ru=ggml_mul_mat_id(ctx,u,x,ids);auto * rh=ggml_swiglu_clamp(ctx,rg,ru,limit);auto * ref=ggml_mul_mat_id(ctx,d,rh,ids);
 auto * nh=llama_exp::build_cpu_gate_up_clamp(ctx,x,ids,g,u,&limit);if(!nh){printf("NULL %s/%s\n",ggml_type_name(gtype),ggml_type_name(utype));return false;}auto * out=ggml_mul_mat_id(ctx,d,nh,ids);
 auto * gf=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(gf,ref);ggml_build_forward_expand(gf,out);ggml_set_output(ref);ggml_set_output(out);ggml_set_output(rh);ggml_set_output(nh);
 ggml_backend_t bs[]={cpu};auto * sched=ggml_backend_sched_new(bs,nullptr,1,128,false,true);if(!ggml_backend_sched_alloc_graph(sched,gf))return false;
 bool ok=true;std::vector<float>a(width*4),b(width*4),ha(hidden*4),hb(hidden*4);
 for(int pattern=0;pattern<3 && ok;++pattern)for(int mask=0;mask<16 && ok;++mask){int32_t ii[4]={0,1,2,3};if(pattern==1){ii[1]=0;ii[3]=2;}if(pattern==2)for(auto &i:ii)i=1;for(int j=0;j<4;++j)if(!(mask&(1<<j)))ii[j]=-1;ggml_backend_tensor_set(ids,ii,0,sizeof(ii));if(ggml_backend_sched_graph_compute(sched,gf)!=GGML_STATUS_SUCCESS){ok=false;break;}
 ggml_backend_tensor_get(ref,a.data(),0,a.size()*4);ggml_backend_tensor_get(out,b.data(),0,b.size()*4);ggml_backend_tensor_get(rh,ha.data(),0,ha.size()*4);ggml_backend_tensor_get(nh,hb.data(),0,hb.size()*4);
 if(memcmp(a.data(),b.data(),a.size()*4)||memcmp(ha.data(),hb.data(),ha.size()*4)){int fi=0;while(fi<int(a.size())&&a[fi]==b[fi])++fi;printf("FAIL %s/%s/%s t%d width%d hidden%d pattern%d mask%d outputdiff%d\n",ggml_type_name(gtype),ggml_type_name(utype),ggml_type_name(dtype),threads,width,hidden,pattern,mask,fi);for(int i=0;i<int(ha.size());++i)if(ha[i]!=hb[i]){printf("HIDDEN %d %a != %a\n",i,ha[i],hb[i]);break;}ok=false;}
 }
 printf("%s %s/%s/%s t%d width%d hidden%d 48 routes\n",ok?"PASS":"FAIL",ggml_type_name(gtype),ggml_type_name(utype),ggml_type_name(dtype),threads,width,hidden);fflush(stdout);
 ggml_backend_sched_free(sched);ggml_backend_buffer_free(wb);ggml_free(ctx);ggml_free(w);return ok;
}
int main(){auto * cpu=ggml_backend_cpu_init();if(!cpu)return 2;
 for(int t:{1,4,16})for(auto ty:{GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0})if(!test(cpu,ty,ty,ty,512,256,t,false))return 1;
 if(!test(cpu,GGML_TYPE_Q4_K,GGML_TYPE_Q3_K,GGML_TYPE_Q8_0,512,256,16,false))return 1;
 if(!test(cpu,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_XXS,GGML_TYPE_Q3_K,2304,5120,16,false))return 1;
 ggml_backend_free(cpu);puts("PASS exact gate-up activations and final FFN outputs; duplicate, empty and partitioned routes");}
