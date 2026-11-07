#!/usr/bin/env python3
"""Host-only page advice parity for real llama_prefetch_rows bodies.

No model files, inference, CUDA execution, system changes or disk preload.
Do not claim that this proves I/O throughput or latency improvements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

KIT = Path(__file__).resolve().parents[1]


def extract(source: str) -> str:
    prefix = 'void llama_prefetch_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {'
    if source.count(prefix) != 1:
        raise ValueError('Expected exactly one upstream llama_prefetch_rows()')
    start = source.index(prefix)
    depth = 0
    for pos in range(source.index('{', start), len(source)):
        if source[pos] == '{':
            depth += 1
        elif source[pos] == '}':
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    raise ValueError('Unterminated prefetch function')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(cmd: list[str], env: dict, path: Path) -> dict:
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=180)
    path.write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f'{cmd[0]} failed exit={p.returncode}: {p.stderr[-1500:]}')
    return json.loads(p.stdout.splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    baseline, candidate = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True)
    out = args.output.resolve()
    if baseline == candidate or any(out.is_relative_to(x) for x in (baseline, candidate)):
        parser.error('sources and output must be separate')
    if out.exists() or not out.parent.is_dir():
        parser.error('output must be a new directory')
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error('C++ compiler unavailable')
    template = (KIT/'tests_cpp/prefetch_rows_fixture.cpp.in').read_text()
    flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror']
    if args.sanitize:
        flags = ['-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                 '-fsanitize=address,undefined', '-fno-sanitize-recover=all']
    report = {'schema_version':1, 'status':'RUNNING',
              'scope':'Source-extracted host page advice parity only; no actual madvise, NVMe or GPU testing',
              'cases':[], 'source_hashes':{}}
    out.mkdir(mode=0o700)
    try:
        for label, tree in [('baseline', baseline), ('candidate', candidate)]:
            source = tree/'src/llama-impl.cpp'
            report['source_hashes'][label] = sha(source)
            function = extract(source.read_text())
            fixture = out/(label+'.cpp')
            fixture.write_text(template.replace('@@PREFETCH_FUNCTION@@', function))
            binary = out/(label+'.bin')
            compiled = subprocess.run([compiler,*flags,str(fixture),'-o',str(binary)],
                                      capture_output=True, text=True, timeout=180)
            (out/(label+'-compile.log')).write_text(compiled.stdout + compiled.stderr)
            if compiled.returncode:
                raise RuntimeError(label+' compile failed: '+compiled.stderr[-1000:])
            profiles = ['unset','0','1','invalid']
            for profile in profiles:
                env = dict(os.environ)
                env.pop('LLAMA_EXPERIMENTAL_PREFETCH_UNIQUE', None)
                if args.sanitize:
                    env['LLAMA_PREFETCH_TEST_QUICK'] = '1'
                if profile != 'unset':
                    env['LLAMA_EXPERIMENTAL_PREFETCH_UNIQUE'] = 'invalid' if profile == 'invalid' else profile
                case = invoke([str(binary)], env, out/(label+'-'+profile+'.log'))
                report['cases'].append({'source':label,'profile':profile,**case})
        if len({x['cases'] for x in report['cases']}) != 1:
            raise RuntimeError('Mismatch in tested case counts')
        if not all(x['status']=='PASS' for x in report['cases']):
            raise RuntimeError('Fixture failed')
        orig = [x for x in report['cases'] if x['source']=='baseline']
        cand = [x for x in report['cases'] if x['source']=='candidate']
        if any(x['input_ranges']!=orig[0]['input_ranges'] for x in orig):
            raise RuntimeError('Baseline changed with env var')
        for x in cand:
            if x['profile'] != '1' and x['input_ranges'] != orig[0]['input_ranges']:
                raise RuntimeError('Candidate not disabled by '+x['profile'])
        if not next(x for x in cand if x['profile']=='1')['input_ranges'] < orig[0]['input_ranges']:
            raise RuntimeError('Unique planner did not reduce the tested input ranges')
        report['status']='PASS_HOST_PAGE_PARITY'
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        report.update(status='FAIL', error=str(exc))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'result':str(out/'result.json')}))
    return 0 if report['status']=='PASS_HOST_PAGE_PARITY' else 1

if __name__=='__main__':
    raise SystemExit(main())
