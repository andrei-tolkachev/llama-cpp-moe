#!/usr/bin/env python3
"""P11 strict host-only worker metadata regression. No model/GPU/network."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('modules/moe/upload_worker.hpp', 'modules/moe/upload_pipeline.hpp',
         'modules/moe/upload_types.hpp', 'tests_cpp/moe_worker_metadata_fixture.cpp')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--sanitizers',action='store_true')
    args=parser.parse_args()
    try:
        source=args.source_root.resolve(strict=True)
        compiler=shutil.which(args.compiler)
        if not compiler: raise ValueError('C++ compiler unavailable')
        if args.output is not None:
            output=args.output.resolve()
            if output.exists() or args.output.is_symlink() or output.is_relative_to(source) or not output.parent.is_dir():
                raise ValueError('output must be new outside source')
            output.mkdir(mode=0o700,parents=False)
            return run(source,output,compiler,args.sanitizers)
        with tempfile.TemporaryDirectory(prefix='llama-p11-host-') as d:
            return run(source,Path(d),compiler,args.sanitizers)
    except (OSError,ValueError,subprocess.TimeoutExpired) as e:
        print(json.dumps({'status':'ERROR','reason':str(e)}))
        return 2

def run(root,output,compiler,sanitizers):
    before={n:sha(root/n) for n in FILES}
    checks=[]
    for san in ([False,True] if sanitizers else [False]):
        executable=output/('p11_sanitized' if san else 'p11_plain')
        cmd=[compiler,'-std=c++17','-O1' if san else '-O2','-Wall','-Wextra','-Werror',
             '-pedantic','-pthread','-I'+str(root),str(root/'tests_cpp/moe_worker_metadata_fixture.cpp'),
             '-o',str(executable)]
        if san:
            cmd+=['-DP11_SKIP_ALLOC_ASSERT','-g','-fno-omit-frame-pointer',
                  '-fsanitize=address,undefined','-fno-sanitize-recover=all']
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=50)
        checks.append({'kind':'compile','sanitizers':san,'returncode':proc.returncode,
                       'stderr':proc.stderr[:1200]})
        if proc.returncode:break
        env=dict(os.environ)
        if san:
            env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
            env['UBSAN_OPTIONS']='halt_on_error=1'
        proc=subprocess.run([str(executable)],capture_output=True,text=True,env=env,timeout=30)
        checks.append({'kind':'run','sanitizers':san,'returncode':proc.returncode,
                       'stdout':proc.stdout[:1200],'stderr':proc.stderr[:1200]})
        if proc.returncode:break
    after={n:sha(root/n) for n in FILES}
    ok=len(checks)==(4 if sanitizers else 2) and all(x['returncode']==0 for x in checks) and before==after
    report={'schema':'llama.experimental.p11.metadata.v1','status':'PASS' if ok else 'FAIL',
            'scope':'P11 C++17 host control metadata only; CUDA/model/token rates not tested',
            'checks':checks,'sources_unchanged':before==after,'file_sha256':before}
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(checks)}))
    return 0 if ok else 1

if __name__=='__main__':sys.exit(main())
