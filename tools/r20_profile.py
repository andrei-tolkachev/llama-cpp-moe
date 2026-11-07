#!/usr/bin/env python3
"""Single diagnostic server. Target CPU/fault/IO evidence is sampled separately from primary A/B."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time

import p17_smoke as smoke
import r20_bench as bench

def read_kv(path):
    return {k: int(v.split()[0]) for line in path.read_text().splitlines()
            if ':' in line for k,v in [line.split(':',1)] if v.split() and v.split()[0].isdigit()}

def snap(pid, expected_alias=None):
    root = Path('/proc')/str(pid)
    result = {k: None for k in ('userspace_ticks','kernel_ticks','minor_faults','major_faults',
                                 'read_bytes','write_bytes','rss_kib','peak_rss_kib')}
    errors = {}
    result.update(monotonic_s=time.monotonic(), unavailable=errors)
    if expected_alias is not None:
        try:
            argv = (root/'cmdline').read_bytes().split(b'\0')
            if not any(a == b'--alias' and b == expected_alias.encode() for a, b in zip(argv, argv[1:])):
                raise ValueError('PID does not identify the isolated server/alias in this procfs namespace')
        except (OSError, ValueError) as exc:
            errors['pid_identity'] = str(exc)
            return result
    try:
        # comm may contain spaces or parentheses. After its final ')' comes field #3.
        stat = (root/'stat').read_text().rsplit(')',1)[1].split()
        result.update(userspace_ticks=int(stat[11]), kernel_ticks=int(stat[12]),
                      minor_faults=int(stat[7]), major_faults=int(stat[9]))
    except (OSError, IndexError, ValueError) as exc:
        errors['stat'] = str(exc)
    for name, fields in (('io', {'read_bytes':'read_bytes','write_bytes':'write_bytes'}),
                         ('status', {'rss_kib':'VmRSS','peak_rss_kib':'VmHWM'})):
        try:
            values = read_kv(root/name)
            result.update({k:values.get(field) for k,field in fields.items()})
        except OSError as exc:
            errors[name] = str(exc)
    return result

def change(before, after):
    keys=['monotonic_s','userspace_ticks','kernel_ticks','minor_faults','major_faults','read_bytes','write_bytes']
    d={k:after[k]-before[k] if None not in (after[k],before[k]) else None for k in keys}
    d['cpu_s_all_threads']=(None if None in (d['userspace_ticks'],d['kernel_ticks']) else
                            (d['userspace_ticks']+d['kernel_ticks'])/os.sysconf('SC_CLK_TCK'))
    return d

def traces(path):
    data=[]
    for line in path.read_text(errors='replace').splitlines():
        marker = next((m for m in ('R20_ENGRAM_JSON', 'P15C_MMQ_SELECT_JSON') if m in line), None)
        if marker:
            fields = line.split(marker, 1)[1].strip().lstrip(':').strip()
            try:
                data.append({'message':line, 'marker':marker, 'fields':json.loads(fields)})
            except ValueError:
                data.append({'message':line, 'marker':marker, 'parse_error':True})
            continue
        if 'experimental_' not in line:continue
        fields=dict(re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)=([^\s,]+)',line))
        data.append({'message':line,'fields':fields})
    return data

def attach(stack, argv, logfile):
    if not shutil.which(argv[0]):return {'cmd':argv,'available':False}
    log=stack.enter_context(logfile.open('xb'))
    proc=subprocess.Popen(argv,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    result={'cmd':argv, 'available':True}
    def stop():
        if proc.poll() is None: proc.send_signal(signal.SIGINT)
        try:proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait()
        result['exit']=proc.returncode
    stack.callback(stop)
    return result

def main():
    profiles={p['name']:p['environment'] for p in json.loads((smoke.KIT/'profiles/experiments.json').read_text())['profiles']}
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--builds',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',choices=list(profiles)+['r19-full','r20-full'],default='planner_only')
    p.add_argument('--ctx',type=int,default=131072);p.add_argument('--predict',type=int,default=256)
    p.add_argument('--ngl',type=int,default=18);p.add_argument('--moe-cache-mib',type=int,default=12288)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--perf',action='store_true',help='sample on-CPU stacks of only the new server after first token')
    p.add_argument('--devices',action='store_true',help='collect whole-host iostat and all-GPU dmon for this diagnostic')
    a=p.parse_args();a.mtp=False;a.fill_tokens=0;a.prompt_file=None
    bins=smoke.validate_inputs(p,a)
    smoke.new_output(p,a.out)
    result={'status':'RUNNING','scope':'one diagnostic request, no warmup; separate from throughput results',
            'host':platform.uname()._asdict(),'tools':{x:shutil.which(x) for x in ('perf','pidstat','iostat','nvidia-smi','nsys','ncu','numactl')},
            'cpuinfo':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'meminfo_kib':read_kv(Path('/proc/meminfo'))}
    out=a.out/'profile.json'
    try:
        result['build']=bench.provenance(a)
        env=(bench.bundle_flags(profiles,include_r20=a.profile=='r20-full') if a.profile in ('r19-full','r20-full') else profiles[a.profile].copy())
        env.update(LLAMA_EXPERIMENTAL_MOE_TRACE='1',LLAMA_EXPERIMENTAL_ENGRAM_TRACE='1')
        if a.profile.endswith('-full'):env['LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE']='1'
        with smoke.server(bins['candidate'],'diag',env,a) as (port,launch):
            off=launch.get('offload_layers')
            if a.ngl and (not off or off[0]!=min(a.ngl,off[1])): raise ValueError(f'Unexpected offload {off} for requested {a.ngl}')
            pid=launch['pid']; ids=bench.prompt_ids(port,a)
            cmd=launch['command']; alias=cmd[cmd.index('--alias')+1]
            with ExitStack() as stack:
                if a.devices:
                    result['iostat']=attach(stack,['iostat','-dxm','-y','1'],a.out/'iostat.log')
                    result['dmon']=attach(stack,['nvidia-smi','dmon','-s','putm','-d','1'],a.out/'gpu-dmon.log')
                result['before']=snap(pid,alias)
                def first():
                    result['first']=snap(pid,alias)
                    if a.perf and 'pid_identity' in result['first']['unavailable']:
                        result['perf']={'attached':False,'reason':'Could not verify procfs PID/alias'}
                    elif a.perf: result['perf']=attach(stack,['perf','record','-F','97','-g','-p',str(pid),'-o',str(a.out/'perf.data')],a.out/'perf.log')
                result['response']=bench.completion_stream(port,ids,a.predict,a.timeout,on_first=first)
                result['end']=snap(pid,alias)
                result['request_delta']=change(result['before'],result['end'])
                result['after_first_observed_delta']=change(result['first'],result['end'])
        if a.perf and (a.out/'perf.data').exists():
            with (a.out/'perf-report.txt').open('w') as f:
                cmd=['perf','report','--stdio','--no-children','--sort','dso,symbol','-i',str(a.out/'perf.data')]
                result['perf_report_exit']=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        result['status']='DIAGNOSTIC_COMPLETE'
    except (OSError,ValueError,KeyError,RuntimeError,KeyboardInterrupt) as e:
        result.update(status='FAIL',error=str(e) or type(e).__name__)
    finally:
        if (a.out/'diag.log').exists():result['trace_messages']=traces(a.out/'diag.log')
        bench.atomic_write(out,result)
    print(result['status'],result.get('error',''),out,flush=True)
    return int(result['status']!='DIAGNOSTIC_COMPLETE')

if __name__=='__main__':raise SystemExit(main())
