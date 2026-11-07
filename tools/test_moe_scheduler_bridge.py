#!/usr/bin/env python3
"""Compile exact P04/P09B/P12/P14A/P14B source snippet against mock events."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(argv,env=None):
    t=time.monotonic()
    try:
        p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=90,env=env)
        return {'returncode':p.returncode,'seconds':round(time.monotonic()-t,3),
                'stdout':p.stdout[-1000:],'stderr':p.stderr[-3500:]}
    except subprocess.TimeoutExpired:
        return {'returncode':124,'seconds':round(time.monotonic()-t,3),'stderr':'TIMEOUT'}

def extract(src,start,stop):
    if src.count(start)!=1:raise ValueError('missing/duplicate exact source anchor: '+start[:65])
    a=src.index(start);b=src.find(stop,a+len(start))
    if b<=a:raise ValueError('end anchor missing: '+stop[:65])
    return src[a:b]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--sanitizers',action='store_true')
    p.add_argument('--compiler',default='c++')
    args=p.parse_args()
    cand=args.candidate.resolve()
    compiler=shutil.which(args.compiler)
    if not cand.is_dir() or not compiler:p.error('candidate/host compiler missing')
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(cand) or
                       args.output.resolve().is_relative_to(ROOT)):
        p.error('output must be new and outside source/candidate')
    m=json.loads((ROOT/'manifest.json').read_text())
    target_path='src/llama-moe-cache.cpp'
    hashes=[entry['after_sha256'] for patch in m['patches'] for entry in patch['targets']
            if entry['path']==target_path]
    target=cand/target_path
    if not hashes or not target.is_file() or sha(target)!=hashes[-1]:
        p.error('target source not the exact final MoE postimage')
    for vendor in ('upload_pipeline.hpp','upload_worker.hpp','upload_types.hpp',
                   'upload_plan.hpp','upload_scheduler.hpp','upload_completion.hpp'):
        local=cand/'src/llama-experimental'/vendor
        original=ROOT/'modules/moe'/vendor
        if not local.is_file() or sha(local)!=sha(original):p.error('vendor header missing/tampered: '+vendor)
    src=target.read_text()
    helper=extract(src,'struct moe_cache_upload_stats {','// gate, up, down or gate_up, down')
    snippet=extract(src,'        std::optional<llama_exp::moe_fill_plan> shared_plan;',
                         '        stats & st = ids.size()')
    if snippet.count('moe_upload_scheduler::assess(')!=1 or snippet.count('moe_upload_pipeline::execute(')!=1 or \
            snippet.count('moe_upload_abort_scope<')!=1:
        p.error('P14 or P09B integration anchor missing')
    record=extract(src,'        // P14B: event is queued AFTER the final slot-map write',
                        '        if (trace.enabled) {')
    if record.count('completion->record_after_slot_map()')!=1 or record.count('abort_completion.commit()')!=1:
        p.error('P14B event record/commit pair missing')
    recovery=extract(src,'        if (!g.upload_complete) {','        size_t n_hit = 0;')
    commit=extract(src,'        g.upload_complete = true;', '        return true;')
    tpl=(ROOT/'tests_cpp/moe_scheduler_bridge_fixture.cpp.in').read_text()
    if (tpl.count('@@P04_HELPER@@')!=1 or tpl.count('@@P09B_COPY_BLOCK@@')!=1 or \
            tpl.count('@@P14B_RECORD_BLOCK@@')!=1):
        p.error('invalid P14B mock fixture markers')
    tpl=tpl.replace('@@RECOVERY@@',recovery).replace('@@COMMIT@@',commit)
    code=tpl.replace('@@P04_HELPER@@',helper).replace('@@P09B_COPY_BLOCK@@',snippet).replace('@@P14B_RECORD_BLOCK@@',record)
    with tempfile.TemporaryDirectory(prefix='llama-p14-bridge-') as td:
        out=args.output.resolve() if args.output else Path(td)
        if args.output:out.mkdir(mode=0o700)
        cpp=out/'moe_scheduler_bridge.cpp';cpp.write_text(code)
        steps=[]
        for mode in ('plain','sanitizers') if args.sanitizers else ('plain',):
            exe=out/f'moe_bridge_{mode}'
            argv=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pthread',
                  '-I'+str(cand/'src'),'-O0' if mode=='sanitizers' else '-O2',str(cpp),'-o',str(exe)]
            if mode=='sanitizers':argv+=['-g','-fno-omit-frame-pointer',
               '-fsanitize=address,undefined','-fno-sanitize-recover=all']
            item={'mode':mode,'compile':run(argv),'runs':[]}
            if item['compile']['returncode']==0:
                for pipeline in (False,True):
                    for shared in (False,True):
                        for scheduler in (False,True):
                            for persistent in (False,True):
                                for events in (False,True):
                                    env=dict(os.environ)
                                    env['LLAMA_EXPERIMENTAL_MOE_PIPELINE']='1' if pipeline else '0'
                                    env['LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN']='1' if shared else '0'
                                    env['LLAMA_EXPERIMENTAL_MOE_SCHEDULER']='1' if scheduler else '0'
                                    env['LLAMA_EXPERIMENTAL_MOE_PERSISTENT']='1' if persistent else '0'
                                    env['LLAMA_EXPERIMENTAL_MOE_EVENT_FENCE']='1' if events else '0'
                                    if mode=='sanitizers':
                                        env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
                                        env['UBSAN_OPTIONS']='halt_on_error=1'
                                    item['runs'].append({'profile':{'pipeline':pipeline,'shared':shared,'scheduler':scheduler,'persistent':persistent,'events':events},
                                        **run([str(exe),'enabled' if pipeline else 'disabled',
                                                'shared' if shared else 'baseline',
                                                'scheduler' if scheduler else 'no-scheduler','event-on' if events else 'event-off'],env)})
            steps.append(item)
            if item['compile']['returncode'] or any(x['returncode'] for x in item['runs']):break
        passed=len(steps)==(2 if args.sanitizers else 1) and all(
            x['compile']['returncode']==0 and len(x['runs'])==32 and
            all(y['returncode']==0 for y in x['runs']) for x in steps)
        report={'schema':'llama.experimental.p14b.bridge.v1','status':'PASS' if passed else 'FAIL',
            'scope':'exact source-extracted P14A/P14B source + host events; not full TU, CUDA or GPU',
            'target_sha256':sha(target),'generated_sha256':sha(cpp),'steps':steps}
        if args.output:(out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'status':report['status'],'target_sha256':report['target_sha256'],
                          'runs':[{'mode':x['mode'],'compile':x['compile']['returncode'],
                          'passing_runs':sum(y['returncode']==0 for y in x['runs']),
                          'failed':[{'profile':y['profile'],'error':y.get('stderr','')[-350:]}
                                    for y in x['runs'] if y['returncode']!=0]} for x in steps]},ensure_ascii=False))
        return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
