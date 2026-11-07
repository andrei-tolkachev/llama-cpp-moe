#!/usr/bin/env python3
"""Launch an isolated llama.cpp server with an explicit measured MoE profile."""
import argparse,json,os,resource
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--profile',type=Path,required=True);p.add_argument('--port',type=int,default=8081)
p.add_argument('--host',default='127.0.0.1');p.add_argument('--print',action='store_true',dest='print_only')
a=p.parse_args();c=json.loads(a.profile.read_text())
if not 1<=a.port<=65535 or not a.binary.is_file() or not a.model.is_file():p.error('Invalid port, binary or model')
if c['parallel']!=1 or c['spec_type']!='none' or c['kv']!='f16':p.error('Selected profiles require one slot, no speculative decoding and F16 KV')
cmd=[str(a.binary.resolve()),'-m',str(a.model.resolve()),'--host',a.host,'--port',str(a.port),
 '--alias',c['profile'],'-np','1','-c',str(c['ctx']),'--fit','off','-ngl',str(c['ngl']),
 '-b',str(c['batch']),'-ub',str(c['ubatch']),'-t',str(c['threads']),'--cpu-mask',c['cpu_mask'],
 '--cpu-strict','1','--cache-type-k','f16','--cache-type-v','f16','--no-warmup','--no-context-shift',
 '--offline','--spec-type','none','--cpu-moe','--moe-cache-mib',str(c['cache_mib'])]
env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_EXPERIMENTAL_','LLAMA_ARG_','GGML_'))}
env.update(c['environment'])
if a.print_only:print(json.dumps({'command':cmd,'environment':c['environment']},indent=2))
else:
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.execvpe(cmd[0],cmd,env)
