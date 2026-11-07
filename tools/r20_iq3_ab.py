#!/usr/bin/env python3
"""Test the built GGML public CPU kernel with opt-in signmask off/on. No model dependency."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path, help='pinned llama.cpp tree for GGML headers')
    p.add_argument('--bin', required=True, type=Path, help='candidate build/bin with libggml-cpu.so')
    p.add_argument('--out', required=True, type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--expert-slice', type=Path, help='isolated bytes of the blk.15 gate expert 0')
    group.add_argument('--target-shard', type=Path, help='shard 3 of the referenced seven-file set')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lib = a.bin.resolve()
    if a.out.exists(): p.error('Output already exists')
    report = {'scope':'single CPU core, GGML registered IQ3_XXS x Q8_K dot, deterministic packed input; ns/row',
              'cpu':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),platform.machine()),
              'library_sha256':hashlib.sha256((lib/'libggml-cpu.so').read_bytes()).hexdigest(),'cases':[]}
    with tempfile.TemporaryDirectory() as tmp:
        prog = Path(tmp)/'dot'
        subprocess.run(['c++','-O2','-std=c++17','-I'+str(a.source.resolve()/'ggml/include'),
                        str(root/'tests_cpp/r20_iq3_dot.cpp'),'-L'+str(lib),'-lggml-cpu','-lggml-base',
                        '-Wl,-rpath,'+str(lib),'-o',str(prog)],check=True)
        if hasattr(os,'sched_getaffinity'):
            cpu=min(os.sched_getaffinity(0)); os.sched_setaffinity(0,{cpu}); report['logical_cpu']=cpu
        extra = []
        cases = [(256,8192,16),(4096,4096,12),(4096,49152,2)]
        if a.expert_slice or a.target_shard:
            ref=json.loads((root/'reports/EXPERT_SLICE.json').read_text())
            with (a.expert_slice or a.target_shard).open('rb') as f:
                if a.target_shard: f.seek(ref['offset'])
                data=f.read(ref['bytes'])
            if hashlib.sha256(data).hexdigest()!=ref['sha256']: p.error('Target expert fingerprint mismatch')
            source_weights=Path(tmp)/'target.weights'
            source_weights.write_bytes(data);extra=[str(source_weights)]
            cases=[(5120,2304,24)]
            report.update(weight_source=ref, activations='deterministic synthetic Q8_K')
        for width, rows, passes in cases:
            result={'width':width,'rows':rows,'passes':passes,'packed_weights_MiB':rows*width/256*98/2**20,'runs':[]}
            report['cases'].append(result)
            for i in range(16):
                on = (0,1,1,0,1,0,0,1)[i%8]
                out=Path(tmp)/'outputs.f32'
                proc=subprocess.run([str(prog),str(width),str(rows),str(passes),str(out),*extra],capture_output=True,
                                    text=True,check=True,env={**os.environ,'LD_LIBRARY_PATH':str(lib),
                                    'LLAMA_EXPERIMENTAL_CPU_IQ3_SIGNMASK':str(on)})
                if on and 'entered x86 IQ3_XXS' not in proc.stderr: p.exit(3,'CPU backend lacks the required ISA, no comparison\n')
                digest=hashlib.sha256(out.read_bytes()).hexdigest()
                if digest != result.setdefault('all_float_outputs_sha256',digest): raise RuntimeError('Bitwise parity failed')
                result['runs'].append({'on':on,'ns_per_row':float(proc.stdout),'stderr':proc.stderr.strip()})
            def summary(flag):
                v=[r['ns_per_row'] for r in result['runs'] if r['on']==flag]
                return {'n':len(v),'median':statistics.median(v),'p95_nearest_rank':sorted(v)[math.ceil(.95*len(v))-1],
                        'sample_variance':statistics.variance(v)}
            result['off']=summary(0);result['on']=summary(1)
            result['latency_reduction_pct']=100*(1-result['on']['median']/result['off']['median'])
    report['status']='BITWISE_EQUAL_REGISTERED_KERNEL'
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')

if __name__=='__main__': main()
