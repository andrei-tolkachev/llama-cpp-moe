#!/usr/bin/env python3
"""P15C host-only source-extracted probe/dispatch and JSONL reporting tests.

Does NOT compile nvcc, execute GPU code, or measure MMQ kernel time.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/p15b_upstream_mmq.cu'
PATCH_B=ROOT/'patches/0014-cuda-mmq-moe-tile-cache.patch'
PATCH_C=ROOT/'patches/0015-cuda-mmq-selection-host-probe.patch'
TEMPLATE=ROOT/'tests_cpp/mmq_tile_cache_fixture.cpp.in'
MODULE=ROOT/'modules/cuda/mmq_select_probe.hpp'
HEADER_B=ROOT/'modules/cuda/mmq_tile_cache.hpp'
FIXTURE=ROOT/'tests_cpp/mmq_select_probe_fixture.cpp'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(argv, *, cwd=None, env=None, timeout=180):
    p=subprocess.run(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"return={p.returncode} command={argv}\nstdout:{p.stdout[-2000:]}\nstderr:{p.stderr[-3000:]}")
    return p


def cxx(src:Path, out:Path, *, compiler:str, sanitize:bool):
    cmd=[compiler,'-std=c++17','-O1','-g','-Wall','-Wextra','-Werror',
         '-pthread','-I',str(MODULE.parent)]
    if sanitize:
        cmd+=['-fsanitize=address,undefined','-fno-omit-frame-pointer',
              '-fno-sanitize-recover=all']
    return invoke([*cmd,str(src),'-o',str(out)])


def run_case(binary:Path, env_flags:dict, args:list[str]=None):
    env=os.environ.copy()
    env.update(env_flags)
    env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
    env['UBSAN_OPTIONS']='halt_on_error=1'
    return invoke([str(binary),*(args or [])],env=env)


def report(log:str)->dict:
    import sys
    sys.path.insert(0,str(ROOT/'tools'))
    try:
        from analyze_p15c_trace import accumulate
        return accumulate(log.splitlines())
    finally:
        sys.path.remove(str(ROOT/'tools'))


def execute(work:Path, compiler:str, sanitize:bool):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['schema_version'] == 1
    p14,p15=manifest['patches'][13:15]
    assert p14['file']==PATCH_B.name and p15['file']==PATCH_C.name
    assert sha(BASE)==p14['targets'][0]['before_sha256']
    assert sha(PATCH_B)==p14['sha256'] and sha(PATCH_C)==p15['sha256']
    assert sha(HEADER_B)==manifest['additional_files'][11]['sha256']
    assert sha(MODULE)==manifest['additional_files'][12]['sha256']
    assert invoke(['git','hash-object',str(BASE)]).stdout.strip()=='49e0ce9050c021f4ffa03924f7bc510f4a96cc41'
    source=work/'ggml/src/ggml-cuda/mmq.cu'
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_bytes(BASE.read_bytes())
    for patch,entry in [(PATCH_B,p14),(PATCH_C,p15)]:
        invoke(['git','apply','--check','--whitespace=error-all',str(patch)],cwd=work)
        invoke(['git','apply','--whitespace=error-all',str(patch)],cwd=work)
        assert sha(source)==entry['targets'][0]['after_sha256'],patch.name
    full=source.read_text()
    pinned=BASE.read_text()
    begin='static int ggml_cuda_mmq_get_J_best('
    helper='static bool ggml_cuda_mmq_moe_tile_cache_type('
    end='\nvoid ggml_cuda_mul_mat_q('
    assert all(x in full and x in pinned if x==begin else x in full for x in (begin,helper,end))
    original_solver=pinned[pinned.index(begin):pinned.index(end)]
    helper_source=full[full.index(helper):full.index(end)]
    glue_start='        // P15C: CPU-side selection timing only, NOT CUDA execution timing.'
    glue_end='    }\n    GGML_ASSERT(J_best > 0);'
    assert full.count(glue_start)==1 and full.count(glue_end)==1
    glue=full[full.index(glue_start):full.index(glue_end)]
    template=TEMPLATE.read_text().replace('#include "mmq_tile_cache.hpp"',
                '#include "mmq_tile_cache.hpp"\n#include "mmq_select_probe.hpp"')
    generated=template.replace('// ORIGINAL_MMq_SOLVER',original_solver).replace('// P15B_HELPERS',helper_source)
    function='''
struct mock_tensor { ggml_type type; };
static int traced_selector(ggml_type type, bool moe, int64_t columns, int & threads) {
    const mock_tensor tensor{type};
    const mock_tensor * src0 = &tensor;
    int mock_ids = 0;
    const int * ids = moe ? &mock_ids : nullptr;
    const int64_t ncols_opt = columns;
    const bool fallback = false;
    const int cc = 90;
    const size_t smpbo = 131072;
    const ggml_prec prec_src1 = GGML_PREC_Q8;
    int J_best = 0;
    int nthreads_best = -777;
''' + glue + '''
    threads = nthreads_best;
    return J_best;
}
static void check_traced_dispatch_parity() {
    for (ggml_type type : quant_types) {
        for (int ncols : {1,2,3,4,8,16,64,65,127,256,513}) {
            for (bool moe : {false,true}) {
                int reference_nt=-777, candidate_nt=-777;
                const int reference=ggml_cuda_mmq_get_J_best(
                      type, ncols, false, 90, 131072, GGML_PREC_Q8, &reference_nt);
                const int candidate=traced_selector(type,moe,ncols,candidate_nt);
                assert(reference==candidate && reference_nt==candidate_nt);
            }
        }
    }
    for (int i=0;i<100;++i) {
        int nt=0;
        assert(traced_selector(GGML_TYPE_Q3_K,true,64,nt)>0);
    }
}
'''
    generated=generated.replace('int main() {',function+'\nint main() {\n    check_traced_dispatch_parity();')
    cpp=work/'source_extracted.cpp';cpp.write_text(generated)
    binary=work/'source_extracted'
    cxx(cpp,binary,compiler=compiler,sanitize=sanitize)
    checks={}
    for name,flags in [
        ('both_off',{'LLAMA_EXPERIMENTAL_CUDA_MMQ_TILE_CACHE':'0','LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE':'0'}),
        ('profile_no_cache',{'LLAMA_EXPERIMENTAL_CUDA_MMQ_TILE_CACHE':'0','LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE':'1'}),
        ('profile_cache',{'LLAMA_EXPERIMENTAL_CUDA_MMQ_TILE_CACHE':'1','LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE':'1'}),
        ('invalid_profile',{'LLAMA_EXPERIMENTAL_CUDA_MMQ_TILE_CACHE':'1','LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE':'1x'}),
    ]:
        case=run_case(binary,flags)
        assert 'P15B host parity OK' in case.stdout
        if name in ('both_off','invalid_profile'):
            assert 'P15C_MMQ_SELECT_JSON ' not in case.stderr
            checks[name]={'trace':0,'parity':'PASS'}
        else:
            x=report(case.stderr)
            assert x['total_calls']==12*11*2+100
            assert x['dropped_samples']==0
            if name=='profile_no_cache':
                assert x['total_cache_hits']==0
            else:
                assert x['total_cache_hits']>0
            checks[name]={'parity':'PASS','trace_calls':x['total_calls'],
                          'cache_hits':x['total_cache_hits'],'trace_records':x['records']}
    fixture_binary=work/'standalone'
    cxx(FIXTURE,fixture_binary,compiler=compiler,sanitize=sanitize)
    off=run_case(fixture_binary,{'LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE':'0'})
    assert not off.stderr.strip()
    cases={}
    for name,args in [('normal',[]),('overflow',['overflow']),('saturated',['saturate'])]:
        run=run_case(fixture_binary,{'LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE':'1'},args)
        x=report(run.stderr)
        cases[name]={'calls':x['total_calls'],'dropped':x['dropped_samples'],'threads':x['threads']}
        if name=='normal':
            assert x['total_calls']==4096+32+4*64 and x['threads']==5
            assert x['total_cache_hits']==4095+4*63
        elif name=='overflow':
            assert x['dropped_samples']==529-512
        else:
            assert x['total_calls']==2
            assert x['groups'][0]['select_ns']==(2**64)-1
    for patch in (PATCH_C,PATCH_B):
        invoke(['git','apply','--reverse','--check',str(patch)],cwd=work)
        invoke(['git','apply','--reverse',str(patch)],cwd=work)
    assert source.read_bytes()==BASE.read_bytes(), 'reverse patch chain not byte-identical'
    result={'status':'PASS_HOST_SOURCE_EXTRACTED','compiler':compiler,
            'sanitizers':sanitize,'source_sha256_r15':p15['targets'][0]['before_sha256'],
            'source_sha256_r16':p15['targets'][0]['after_sha256'],
            'branch_checks':checks,'probe_checks':cases,
            'not_tested':['nvcc / CUDA compilation','GPU memory or kernel speed','real GGUF parity','model tokens/s']}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',default='g++')
    p.add_argument('--sanitizers',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output:
        path=a.output.resolve()
        if path.exists() or ROOT==path or ROOT in path.parents or not path.parent.is_dir():
            p.error('--output must be a NEW directory outside the source package')
        path.mkdir(mode=0o700)
        execute(path,a.compiler,a.sanitizers)
    else:
        with tempfile.TemporaryDirectory(prefix='llama-p15c-host-') as t:
            execute(Path(t),a.compiler,a.sanitizers)


if __name__=='__main__':main()
