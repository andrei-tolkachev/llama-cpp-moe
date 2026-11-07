#!/usr/bin/env python3
"""Host-only P16A MTP accept/capture parity gate (does not load model or CUDA)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CPP=ROOT/'tests_cpp/mtp_verify_window_fixture.cpp'
HEADER=ROOT/'modules/speculative/mtp_verify_window.hpp'

def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path,help='New directory outside the project')
    parser.add_argument('--compiler',default='g++')
    parser.add_argument('--sanitizers',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or args.output.is_symlink() or output.is_relative_to(ROOT.resolve()) or not output.parent.is_dir():
        parser.error('output must be a new directory outside source, with an existing parent')
    compiler=shutil.which(args.compiler)
    if not compiler:
        parser.error('C++ compiler not available')
    output.mkdir(mode=0o700)
    report={'status':'RUNNING','scope':'HOST_CXX17_MTP_WINDOW_ONLY_NO_GPU_NO_LLAMA_LINK',
        'sanitizers':args.sanitizers,'compiler':compiler,'source_sha256':sha(CPP),'header_sha256':sha(HEADER),'modes':{}}
    result=output/'p16a_result.json'
    def save():result.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        binary=output/'test_p16a_mtp_window'
        cmd=[compiler,'-std=c++17','-O1' if args.sanitizers else '-O2',
             '-Wall','-Wextra','-Werror','-pthread',str(CPP),'-o',str(binary)]
        if args.sanitizers:
            cmd.extend(['-g','-fno-omit-frame-pointer','-fsanitize=address,undefined'])
        build=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
        (output/'build.log').write_text(build.stdout+build.stderr)
        if build.returncode: raise RuntimeError('host C++ compilation failed')
        for mode in ('unset','0','1','1x'):
            env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'}
            if mode=='unset':env.pop('LLAMA_EXPERIMENTAL_MTP_VERIFY_WINDOW',None)
            else:env['LLAMA_EXPERIMENTAL_MTP_VERIFY_WINDOW']=mode
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60,env=env)
            (output/f'{mode}.log').write_text(run.stdout+run.stderr)
            if run.returncode!=0 or 'parity OK' not in run.stdout:
                raise RuntimeError(f'parity failed in env mode: {mode}')
            report['modes'][mode]={'rc':run.returncode,'output':run.stdout.strip()}
            save()
        report['status']='PASS_HOST_ONLY'
    except Exception as exc:
        report['status']='FAIL_HOST_ONLY';report['error']=str(exc)
        save()
        print(str(exc),file=sys.stderr)
        return 1
    save();print(json.dumps(report,ensure_ascii=False))
    return 0

if __name__=='__main__':sys.exit(main())
