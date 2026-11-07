#!/usr/bin/env python3
"""P16B host-only checkpoint/replay state gate; NEVER loads a model or uses CUDA."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CPP=ROOT/'tests_cpp/mtp_rewind_guard_fixture.cpp'
HEADER=ROOT/'modules/speculative/mtp_rewind_guard.hpp'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--compiler',default='g++')
    ap.add_argument('--sanitizers',action='store_true')
    a=ap.parse_args()
    output=a.output.resolve()
    if output.exists() or a.output.is_symlink() or output.is_relative_to(ROOT.resolve()) or not output.parent.is_dir():
        ap.error('output must be a NEW directory OUTSIDE source')
    compiler=shutil.which(a.compiler)
    if not compiler:ap.error('C++ compiler missing')
    output.mkdir(mode=0o700)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result={'status':'RUNNING','scope':'HOST_CXX17_GUARD_ONLY_NOT_NATIVE_LLAMA_OR_KV',
            'compiler':compiler,'sanitizers':a.sanitizers,'header_sha256':sha(HEADER),'fixture_sha256':sha(CPP),'modes':{}}
    def save():(output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        executable=output/'test_guard'
        cmd=[compiler,'-std=c++17','-O1' if a.sanitizers else '-O2','-Wall','-Wextra','-Werror','-pthread',str(CPP),'-o',str(executable)]
        if a.sanitizers:cmd+=['-g','-fno-omit-frame-pointer','-fsanitize=address,undefined','-fno-sanitize-recover=all']
        build=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        (output/'build.log').write_text(build.stdout+build.stderr)
        if build.returncode:raise RuntimeError('C++ build: '+build.stderr[-1500:])
        for mode in ('unset','0','1','1x'):
            env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
            if mode=='unset':env.pop('LLAMA_EXPERIMENTAL_MTP_REWIND_GUARD',None)
            else:env['LLAMA_EXPERIMENTAL_MTP_REWIND_GUARD']=mode
            run=subprocess.run([str(executable)],capture_output=True,text=True,env=env,timeout=45)
            (output/(mode+'.log')).write_text(run.stdout+run.stderr)
            if run.returncode or 'P16B PASS 8275' not in run.stdout:
                raise RuntimeError('host fixture failed mode='+mode+' '+run.stderr[-1500:])
            result['modes'][mode]={'rc':run.returncode,'output':run.stdout.strip()}
            save()
        result['status']='PASS_HOST_ONLY'
    except Exception as exc:
        result.update(status='FAIL_HOST_ONLY',error=str(exc))
        save();print(str(exc),file=sys.stderr);return 1
    save();print(json.dumps(result,ensure_ascii=False));return 0
if __name__=='__main__':sys.exit(main())
