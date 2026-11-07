#!/usr/bin/env python3
"""P14 exact P04 scheduling oracle and deferred mock GPU upload, host-only."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()

def oracle():
    # The ORIGINAL P04 source is immutable; extract its implementation directly.
    patch=(ROOT/'patches/0004-moe-upload-staging.patch').read_text()
    a=patch.index('+struct moe_cache_upload_stats {')
    b=patch.index(' // gate, up, down or gate_up, down',a)
    lines=patch[a:b].splitlines(keepends=True)
    if not all(x.startswith('+') for x in lines):raise RuntimeError('unexpected P04 oracle diff layout')
    return ''.join(x[1:] for x in lines)

def execute(argv,env=None):
    start=time.monotonic()
    try:
        p=subprocess.run(argv,text=True,capture_output=True,env=env,timeout=95)
        return {'returncode':p.returncode,'seconds':round(time.monotonic()-start,3),
                'stdout':p.stdout[-1500:],'stderr':p.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        return {'returncode':124,'seconds':round(time.monotonic()-start,3),'stderr':'TIMEOUT'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--sanitizers',action='store_true')
    ap.add_argument('--compiler',default='c++')
    args=ap.parse_args()
    compiler=shutil.which(args.compiler)
    if compiler is None:ap.error('compiler missing')
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(ROOT)):
        ap.error('output must be new and outside source root')
    with tempfile.TemporaryDirectory(prefix='p14-host-') as td:
        out=args.output.resolve() if args.output else Path(td)
        if args.output:out.mkdir(mode=0o700)
        tpl=(ROOT/'tests_cpp/moe_scheduler_fixture.cpp.in').read_text()
        if tpl.count('@@P04_HELPER@@')!=1:raise RuntimeError('missing single oracle template')
        src=out/'moe_scheduler_fixture.cpp'
        src.write_text(tpl.replace('@@P04_HELPER@@',oracle()))
        modes=['plain','sanitizers'] if args.sanitizers else ['plain']
        results=[]
        for mode in modes:
            bin=out/f'moe_scheduler_{mode}'
            cmd=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pthread',
                 '-I'+str(ROOT/'modules'),'-O1' if mode=='sanitizers' else '-O2',str(src),'-o',str(bin)]
            if mode=='sanitizers':cmd+=['-g','-fno-omit-frame-pointer','-fsanitize=address,undefined','-fno-sanitize-recover=all']
            step={'mode':mode,'compile':execute(cmd),'runs':[]}
            if step['compile']['returncode']==0:
                for sch in (False,True):
                    for persistent in (False,True):
                        env=dict(os.environ)
                        env['LLAMA_EXPERIMENTAL_MOE_SCHEDULER']='1' if sch else '0'
                        env['LLAMA_EXPERIMENTAL_MOE_PERSISTENT']='1' if persistent else '0'
                        if mode=='sanitizers':
                            env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
                            env['UBSAN_OPTIONS']='halt_on_error=1'
                        step['runs'].append({'scheduler':sch,'persistent':persistent,
                                             **execute([str(bin),'enabled' if sch else 'disabled'],env)})
            results.append(step)
            if step['compile']['returncode']!=0 or any(x['returncode']!=0 for x in step['runs']):break
        passed=len(results)==len(modes) and all(x['compile']['returncode']==0 and
                     len(x['runs'])==4 and all(y['returncode']==0 for y in x['runs']) for x in results)
        report={'schema':'llama.experimental.p14.host.v1','status':'PASS' if passed else 'FAIL',
                'p04_oracle_sha256':sha(ROOT/'patches/0004-moe-upload-staging.patch'),
                'scheduler_sha256':sha(ROOT/'modules/moe/upload_scheduler.hpp'),
                'fixture_sha256':sha(src),'cases_per_run':2400,
                'scope':'host-only P04 exact byte/staging estimates, mock deferred backend, no CUDA/device events',
                'steps':results}
        if args.output:(out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'status':report['status'],'scope':report['scope'],
                          'steps':[{ 'mode':x['mode'],'compiled':x['compile']['returncode']==0,
                                     'runs':[(r['scheduler'],r['persistent'],r['returncode']) for r in x['runs']]}
                                      for x in results]},ensure_ascii=False))
        return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
