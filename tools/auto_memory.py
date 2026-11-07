#!/usr/bin/env python3
"""Startup memory planner using llama.cpp's own no-allocation estimator.

This does not resize an active context. Prompt files are raw text; a configurable
headroom accounts for chat formatting. An exact formatted token count can be supplied.
"""
import argparse, json, math, os, re, resource, subprocess
from pathlib import Path

PHASE_FLAG = 'LLAMA_EXPERIMENTAL_MOE_PHASE_PREFILL_MIB'

def phase_memory(decode, prefill):
    # SWA/recurrent state is allocated for the large ubatch and retained at decode.
    return decode['model_mib'] + max(decode['context_mib'], prefill['context_mib']) + decode['compute_mib']

def context_for(required, native, fallback):
    if required is None:
        return min(fallback, native)
    if required > native:
        raise ValueError('Input plus output/headroom exceeds the model context limit')
    if required < 1:
        raise ValueError('Required context must be positive')
    return min(native, max(4096, 1 << (required-1).bit_length()))

def memory_rows(text):
    rows = {}
    for line in text.splitlines():
        m = re.fullmatch(r'(CUDA\d+|Host)\s+(\d+)\s+(\d+)\s+(\d+)\s*', line)
        if m:
            if m[1] in rows:
                raise ValueError('Duplicate memory-estimator device')
            rows[m[1]] = dict(zip(['model_mib','context_mib','compute_mib'],map(int,m.groups()[1:])))
    if set(rows) != {'CUDA0','Host'}:
        raise ValueError('This startup adapter currently requires one CUDA device and one Host row')
    return rows

def choose_cache(cap, available, reserve, measure, step=128):
    if available <= reserve:
        raise ValueError('Insufficient free VRAM for the requested reserve')
    budget = available - reserve
    low, high, best = 0, cap//step, None
    while low <= high:
        mid=(low+high)//2
        size=mid*step
        used=measure(size)
        if not math.isfinite(used) or used < 0:
            raise ValueError('Invalid estimated memory size')
        if used <= budget:
            best=size;low=mid+1
        else:
            high=mid-1
    if best is None or best < step:
        raise ValueError('Requested context and useful expert cache do not fit; no context or precision was reduced')
    return best

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--ctx',default='auto')
    inp=p.add_mutually_exclusive_group()
    inp.add_argument('--prompt-file',type=Path)
    inp.add_argument('--input-tokens',type=int,help='Exact formatted input token count')
    p.add_argument('--max-output',type=int,default=4096)
    p.add_argument('--template-headroom',type=int,default=1024)
    p.add_argument('--reserve-mib',type=int,default=2048)
    p.add_argument('--out',type=Path)
    p.add_argument('--run',action='store_true')
    p.add_argument('--port',type=int,default=8081)
    a=p.parse_args()
    if a.max_output<1 or a.template_headroom<0 or a.reserve_mib<512 or not 1<=a.port<=65535:
        p.error('Invalid output/headroom/reserve/port')
    binary=a.binary.resolve();model=a.model.resolve();fit=binary.with_name('llama-fit-params')
    if not all(x.is_file() for x in [binary,model,fit]):
        p.error('Binary, native memory estimator or model is missing; build the production tools')
    if a.out and a.out.exists():
        p.error('Output plan must be a new file')
    c=json.loads(a.profile.read_text())
    if c['parallel']!=1 or c['spec_type']!='none' or c['kv']!='f16':
        p.error('This adapter preserves the tested single-slot, F16, non-speculative profiles')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in [None,'','0']:
        p.error('GPU remapping needs an explicit device mapping; this adapter uses physical GPU 0')
    # Never compete with a live llama.cpp instance or restart a user service.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:exe=(proc/'exe').resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe.name=='llama-server':
            p.error('An active llama-server exists; plan after it stops. No process was changed.')
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_EXPERIMENTAL_','LLAMA_ARG_'))}
    env.update(c['environment']);env['LD_LIBRARY_PATH']=str(binary.parent)
    phase_cap = int(env.get(PHASE_FLAG, '0'))
    if phase_cap and (phase_cap < 1 or phase_cap > c['cache_mib'] or c['ubatch'] < 512):
        p.error('Invalid phase cache budget or ubatch')
    estimate_env = env.copy()
    estimate_env.pop(PHASE_FLAG, None)
    def free_mib():
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
        rows=[x.split(',') for x in text.strip().splitlines()]
        if len(rows)!=1 or rows[0][0].strip()!='0':
            raise ValueError('This startup adapter requires a single NVIDIA GPU')
        return int(rows[0][1])
    free_before=free_mib()
    if free_before<a.reserve_mib+512:
        p.error('Insufficient free VRAM for safe planning')
    measurements=[]
    estimates = {}
    def estimate(ctx,cache,ubatch=None):
        ubatch = c['ubatch'] if ubatch is None else ubatch
        key = (ctx,cache,ubatch)
        if key in estimates:
            return estimates[key]
        cmd=[str(fit),'-m',str(model),'--fit-print','on','--fit','off','-lv','5',
          '-c',str(ctx),'-np','1','-ngl',str(c['ngl']),'-b',str(c['batch']),'-ub',str(ubatch),
          '-t',str(c['threads']),'--cpu-mask',c['cpu_mask'],'--cpu-strict','1',
          '--cache-type-k','f16','--cache-type-v','f16','--cpu-moe','--moe-cache-mib',str(cache),'--offline']
        q=subprocess.run(cmd,env=estimate_env,capture_output=True,text=True,errors="replace",timeout=120)
        if q.returncode:
            raise RuntimeError('Native memory estimator failed: '+q.stderr[-1200:])
        rows=memory_rows(q.stdout)
        native=re.search(r'n_ctx_train\s*=\s*(\d+)',q.stderr)
        if not native:
            raise ValueError('Native context limit was not reported; refusing to guess')
        measurements.append({'ctx':ctx,'cache_mib':cache,'ubatch':ubatch,'memory':rows})
        estimates[key] = (rows,int(native[1]))
        return estimates[key]
    initial,int_native=estimate(c['ctx'],c['cache_mib'])
    n=a.input_tokens
    if a.prompt_file:
        q=subprocess.run([str(binary.with_name('llama-tokenize')),'-m',str(model),'-f',str(a.prompt_file.resolve()),'--ids','--no-escape'],env=estimate_env,capture_output=True,text=True,errors="replace",timeout=120)
        if q.returncode:
            raise RuntimeError('Native tokenizer failed: '+q.stderr[-1200:])
        ids=json.loads(q.stdout)
        if not isinstance(ids,list) or any(type(x)!=int for x in ids):raise ValueError('Invalid native tokenizer output')
        n=len(ids)
    if n is not None and n<0:p.error('Input token count cannot be negative')
    required=None if n is None else n+a.max_output+(a.template_headroom if a.prompt_file else 0)
    ctx=context_for(required,int_native,c['ctx']) if a.ctx=='auto' else int(a.ctx)
    if ctx<1 or ctx>int_native or (required is not None and required>ctx):
        p.error('Context is outside native bounds or smaller than the request; input will not be truncated')
    available=min(free_before,free_mib())
    def used(cache):
        rows,_=estimate(ctx,cache,16 if phase_cap else c['ubatch'])
        if phase_cap:
            large,_=estimate(ctx,cache,c['ubatch'])
            return phase_memory(rows['CUDA0'],large['CUDA0'])
        return sum(rows['CUDA0'].values())
    cache=choose_cache(c['cache_mib'],available,a.reserve_mib,used)
    final=used(cache)
    phase_plan = None
    if phase_cap:
        def prefill_used(size):
            rows,_=estimate(ctx,size,c['ubatch'])
            return sum(rows['CUDA0'].values())
        prefill_cache=choose_cache(min(phase_cap,cache),available,a.reserve_mib,prefill_used)
        prefill_final=prefill_used(prefill_cache)
        env[PHASE_FLAG]=str(prefill_cache)
        phase_plan={'decode_cache_mib':cache,'prefill_cache_mib':prefill_cache,
          'decode_ubatch':16,'prefill_ubatch':c['ubatch'],
          'decode_estimated_gpu_mib':final,'prefill_estimated_gpu_mib':prefill_final,
          'expert_contents_retained':False,'kv_state_retained':True}
        final=max(final,prefill_final)
    if final+a.reserve_mib>free_mib():
        raise RuntimeError('Free VRAM changed during planning; retry when the GPU is idle')
    cmd=[str(binary),'-m',str(model),'--host','127.0.0.1','--port',str(a.port),
      '--alias',c['profile'],'-np','1','-c',str(ctx),'--fit','off','-ngl',str(c['ngl']),
      '-b',str(c['batch']),'-ub',str(c['ubatch']),'-t',str(c['threads']),'--cpu-mask',c['cpu_mask'],
      '--cpu-strict','1','--cache-type-k','f16','--cache-type-v','f16','--no-warmup',
      '--no-context-shift','--offline','--spec-type','none','--cpu-moe','--moe-cache-mib',str(cache)]
    plan={'scope':'STARTUP_ONLY_NOT_LIVE_KV_RESIZING','native_context':int_native,'ctx':ctx,'cache_mib':cache,
      'input_tokens':n,'required_tokens':required,'auto_context_basis':'input plus output/headroom' if required is not None else 'existing profile default',
      'available_vram_mib':available,'reserve_mib':a.reserve_mib,'estimated_gpu_mib':final,
      'cache_cap_mib':c['cache_mib'],'measurements':measurements,'command':cmd,
      'environment':{k:env[k] for k in c['environment']},'phase_plan':phase_plan,
      'precision_changed':False,'input_truncated':False,'device_scope':'single CUDA GPU; profile-specific CPU settings'}
    if a.out:
        with a.out.open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
    print(json.dumps(plan,indent=2),flush=True)
    if a.run:
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        scope=['systemd-run','--user','--scope','--quiet','--unit=llama-auto-'+str(os.getpid()),
          '--property=MemoryHigh=106G','--property=MemoryMax=108G','--property=MemorySwapMax=4G',
          '--property=TasksMax=256','--']
        os.execvpe(scope[0],scope+cmd,env)

if __name__=='__main__':main()
