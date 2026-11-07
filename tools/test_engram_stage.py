#!/usr/bin/env python3
"""Compile and run opt-in Engram GGML staging's source-independent host core."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--sanitizers',action='store_true')
    args=parser.parse_args()
    root=args.source_root.resolve(strict=True)
    out=args.output.resolve()
    source=root/'tests_cpp/engram_graph_stage_fixture.cpp'
    dependencies=[source,*(root/'modules/engram'/p for p in ['graph_stage.hpp','mapped_rows.hpp','row_cache.hpp'])]
    if out.exists() or args.output.is_symlink() or out.is_relative_to(root) or not out.parent.is_dir():
        parser.error('output must be new and outside source root')
    if not shutil.which(args.compiler): parser.error('compiler unavailable')
    if any(not p.is_file() or p.is_symlink() for p in dependencies): parser.error('invalid source file')
    before={str(p.relative_to(root)):sha(p) for p in dependencies}
    out.mkdir(mode=0o700)
    flags=['-std=c++17','-Wall','-Wextra','-Werror','-pthread','-O2']
    if args.sanitizers:
        flags=['-std=c++17','-Wall','-Wextra','-Werror','-pthread','-O1','-g',
               '-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all']
    binary=out/'engram_stage_fixture'
    command=[shutil.which(args.compiler),*flags,str(source),'-o',str(binary)]
    results={}
    for name,cmd,timeout in [('compile',command,120),('execute',[str(binary)],100)]:
        if name=='execute' and results['compile']['exit_code']!=0:break
        tick=time.monotonic()
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False)
            results[name]={'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,
                           'stderr':p.stderr,'seconds':round(time.monotonic()-tick,3)}
        except subprocess.TimeoutExpired as e:
            results[name]={'command':cmd,'exit_code':-1,'error':'timeout','seconds':timeout}
    try: js=json.loads(results['execute']['stdout'])
    except (KeyError,ValueError):js={}
    status=('PASS' if all(x['exit_code']==0 for x in results.values()) and
            js=={'status':'PASS','scenarios':4,'gpu':False,'model':False} and
            before=={str(p.relative_to(root)):sha(p) for p in dependencies} else 'FAIL')
    report={'status':status,'sanitizers':args.sanitizers,'tests':results,'inputs_sha256':before,
            'model_or_cuda_runtime_executed':False,'graph_path_compiled':False}
    (out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':status,'result':str(out/'result.json')}))
    return 0 if status=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
