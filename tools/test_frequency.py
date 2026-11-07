from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=(a.source/'src/llama-moe-cache.cpp').read_text();body=s[s.index('struct moe_cache_slru {'):s.index('// An opt-in limit')]
header='''#include <algorithm>\n#include <cassert>\n#include <cstdint>\n#include <cstdlib>\n#include <vector>\n#include <random>\n#include <iostream>\n#include <set>\n#define GGML_ASSERT assert\n#define GGML_ABORT(...) std::abort()\n'''
test=r'''
int main() {
  for (bool use_frequency : {false,true}) for (bool reuse : {false,true}) {
    setenv("LLAMA_EXPERIMENTAL_MOE_FREQUENCY",use_frequency?"1":"0",1);
    moe_cache_slru c; c.init(5,31,23);
    std::vector<std::vector<int32_t>> maps(5,std::vector<int32_t>(31,-1));
    for(int l=0;l<5;++l)c.slot_map[l]=maps[l].data();
    std::vector<int32_t> banks(23,-1);std::mt19937 rng(2971);
    for(int step=0;step<50000;++step) {
      if(step%811==0) {c.invalidate_banks();std::fill(banks.begin(),banks.end(),-1);}
      int l=rng()%5, n=rng()%25;std::vector<int32_t> ids;std::set<int32_t> want;
      for(int j=0;j<n;++j){int e=rng()%31;ids.push_back(e);want.insert(e);}
      size_t expected_hits=0;for(int e:want)expected_hits+=maps[l][e]>=0;
      std::vector<moe_cache_slru::fill> fills;size_t hits;
      bool ok=reuse?c.plan_reuse(l,ids.data(),ids.size(),fills,hits):c.plan(l,ids.data(),ids.size(),fills,hits);
      assert(ok==(want.size()<=23));if(!ok)continue;
      assert(hits==expected_hits && fills.size()+hits==want.size());
      for(auto f:fills){assert(want.count(f.expert));banks[f.slot]=l*31+f.expert;}
      for(int e:want){assert(maps[l][e]>=0);assert(banks[maps[l][e]]==l*31+e);}
      for(int k=0;k<23;++k)if(c.key_of[k]>=0){auto key=c.key_of[k];assert(maps[key/31][key%31]==k);assert(banks[k]==key);}
      for(int a=0;a<5;++a)for(int e=0;e<31;++e)if(maps[a][e]>=0)assert(c.key_of[maps[a][e]]==a*31+e);
      if(use_frequency){assert(c.heap.size()==23);for(size_t j=0;j<23;++j){assert(c.heap_pos[c.heap[j]]==int(j));if(j>0)assert(!c.colder(c.heap[j],c.heap[(j-1)/2]));}}
    }
  }
  setenv("LLAMA_EXPERIMENTAL_MOE_FREQUENCY","1",1);
  moe_cache_slru c;c.init(1,5,2);std::vector<int32_t> m(5,-1);c.slot_map[0]=m.data();
  std::vector<moe_cache_slru::fill> f;size_t hits;int32_t hot=0;
  for(int i=0;i<100;++i)c.plan(0,&hot,1,f,hits);
  int32_t cold=1;c.plan(0,&cold,1,f,hits);cold=2;c.plan(0,&cold,1,f,hits);
  assert(m[0]>=0 && m[1]<0 && m[2]>=0);
  int32_t next=3;for(int i=0;i<1024;++i)c.plan(0,&next,1,f,hits);
  cold=4;c.plan(0,&cold,1,f,hits);assert(m[0]<0 && m[3]>=0);
  int32_t both[]={3,4,3};c.plan(0,both,3,f,hits);assert(hits==2);
  int32_t too_many[]={0,1,2};auto old=m;assert(!c.plan(0,too_many,3,f,hits));assert(m==old);
  std::cout<<"PASS 200000 randomized multi-layer requests, map/bank identity, heap invariants, decay drift, duplicates, overflow, invalidation"<<std::endl;
}
'''
a.output.write_text(header+body+test)
