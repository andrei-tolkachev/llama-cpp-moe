#!/usr/bin/env python3
"""P15B source-extracted exact tile-selection parity test (HOST ONLY).

Reconstructs candidate mmq.cu from pinned upstream via the real patch,
extracts the original MMQ J solver and P15B wrapper for a C++17 mock.
It DOES NOT compile nvcc or infer actual CUDA throughput.
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
SOURCE=ROOT/'validation/p15b_upstream_mmq.cu'
PATCH=ROOT/'patches/0014-cuda-mmq-moe-tile-cache.patch'
HEADER=ROOT/'modules/cuda/mmq_tile_cache.hpp'
TEMPLATE=ROOT/'tests_cpp/mmq_tile_cache_fixture.cpp.in'
UPSTREAM_BLOB='49e0ce9050c021f4ffa03924f7bc510f4a96cc41'

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def run(cmd, *, cwd=None,env=None,timeout=120):
    result=subprocess.run(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('failed: '+str(cmd)+'\n'+result.stdout+'\n'+result.stderr)
    return result.stdout.strip()

def execute(path:Path,compiler:str,sanitize:bool):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entry=manifest['patches'][13]
    if entry['file']!=PATCH.name or entry['targets'][0]['before_sha256']!=sha(SOURCE):
        raise RuntimeError('Pinned source does not match manifest')
    if sha(PATCH)!=entry['sha256'] or sha(HEADER)!=manifest['additional_files'][11]['sha256']:
        raise RuntimeError('Patched sources do not match manifest')
    if run(['git','hash-object',str(SOURCE)])!=UPSTREAM_BLOB:
        raise RuntimeError('Pinned git blob differs from upstream')
    src=path/'ggml/src/ggml-cuda/mmq.cu'
    src.parent.mkdir(parents=True)
    src.write_bytes(SOURCE.read_bytes())
    run(['git','apply','--check','--whitespace=error-all',str(PATCH)],cwd=path)
    run(['git','apply','--whitespace=error-all',str(PATCH)],cwd=path)
    if sha(src)!=entry['targets'][0]['after_sha256']:
        raise RuntimeError('Candidate postimage SHA mismatch')
    original=SOURCE.read_text()
    candidate=src.read_text()
    begin='static int ggml_cuda_mmq_get_J_best('
    end='\nvoid ggml_cuda_mul_mat_q('
    assert original.count(begin)==1 and original.count(end)==1
    original_solver=original[original.index(begin):original.index(end)]
    before='static bool ggml_cuda_mmq_moe_tile_cache_type('
    assert candidate.count(before)==1
    cached_wrapper=candidate[candidate.index(before):candidate.index(end)]
    template=TEMPLATE.read_text()
    assert template.count('// ORIGINAL_MMq_SOLVER')==1
    assert template.count('// P15B_HELPERS')==1
    generated=(template.replace('// ORIGINAL_MMq_SOLVER',original_solver)
                .replace('// P15B_HELPERS',cached_wrapper))
    cpp=path/'candidate_parity.cpp';cpp.write_text(generated)
    binary=path/'candidate_parity'
    cmd=[compiler,'-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-pthread',
         '-I',str(HEADER.parent)]
    if sanitize:
        cmd+=['-fsanitize=address,undefined','-fno-sanitize-recover=all',
              '-fno-omit-frame-pointer']
    cmd += [str(cpp),'-o',str(binary)]
    run(cmd,timeout=120)
    reports={}
    for name,flag in [('off','0'),('on','1'),('invalid','1more')]:
        env=os.environ.copy();env['LLAMA_EXPERIMENTAL_CUDA_MMQ_TILE_CACHE']=flag
        if sanitize:
            env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
            env['UBSAN_OPTIONS']='halt_on_error=1'
        reports[name]=run([str(binary)],env=env,timeout=120)
    assert 'cache_enabled=0' in reports['off']
    assert 'cache_enabled=1' in reports['on']
    assert 'cache_enabled=0' in reports['invalid']
    run(['git','apply','--reverse','--check',str(PATCH)],cwd=path)
    run(['git','apply','--reverse',str(PATCH)],cwd=path)
    if src.read_bytes()!=SOURCE.read_bytes():
        raise RuntimeError('Reverse did not restore byte-exact source')
    report={'status':'PASS_HOST_SOURCE_EXTRACTED',
      'sanitizers':sanitize,'upstream_blob':UPSTREAM_BLOB,
      'upstream_sha256':sha(SOURCE),'patch_sha256':sha(PATCH),
      'candidate_sha256':entry['targets'][0]['after_sha256'],
      'results':reports,
      'not_tested':['NVCC compilation/link','GPU kernel behavior','RTX 5090 latency',
                    'GGUF logits/model parity','actual MMQ tile-selection performance']}
    (path/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sanitizers',action='store_true')
    p.add_argument('--compiler',default='c++')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output:
        target=a.output.resolve()
        if target.exists() or target==ROOT or ROOT in target.parents or not target.parent.is_dir():
            p.error('--output must be a new directory outside the source package')
        target.mkdir(mode=0o700)
        execute(target,a.compiler,a.sanitizers)
    else:
        with tempfile.TemporaryDirectory(prefix='llama-p15b-') as tmp:
            execute(Path(tmp),a.compiler,a.sanitizers)

if __name__=='__main__':main()
