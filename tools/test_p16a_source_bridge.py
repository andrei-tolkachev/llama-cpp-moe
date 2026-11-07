#!/usr/bin/env python3
"""Compile P16A's exact patched process()/accept() snippets against a mock ggml host."""
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
PARTS=ROOT/'validation/p16a_upstream_fragments.json'
TEMPLATE=ROOT/'tests_cpp/mtp_source_bridge_fixture.cpp.in'
HEADER=ROOT/'modules/speculative/mtp_verify_window.hpp'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--compiler',default='g++')
    ap.add_argument('--sanitizers',action='store_true')
    opts=ap.parse_args()
    out=opts.output.resolve()
    if out.exists() or opts.output.is_symlink() or out.is_relative_to(ROOT.resolve()) or not out.parent.is_dir():
        ap.error('output must be a NEW directory outside the package')
    compiler=shutil.which(opts.compiler)
    if not compiler:ap.error('compiler not available')
    data=json.loads(PARTS.read_text())
    f=data['fragments']
    assert len(f)==5
    process=f[3]['modified'];accept=f[4]['modified']
    assert process.startswith('        for (llama_seq_id seq_id') and process.endswith('    }\n')
    assert accept.startswith('    void accept(') and accept.endswith('};\n')
    assert process.count('window.retained_rows')==3
    assert accept.count('window.accepted_row(n_accepted)')==1
    code=TEMPLATE.read_text().replace('@@P16_PROCESS@@',process.rstrip()).replace('@@P16_ACCEPT@@',accept.rsplit('};\n',1)[0].rstrip())
    out.mkdir(mode=0o700)
    result={'status':'RUNNING','scope':'SOURCE_EXTRACTED_HOST_ONLY','compiler':compiler,'sanitizers':opts.sanitizers,
            'source_fragment_sha256':sha(PARTS),'header_sha256':sha(HEADER)}
    def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        cpp=out/'source_bridge.cpp';cpp.write_text(code)
        exe=out/'source_bridge'
        cmd=[compiler,'-std=c++17','-O1' if opts.sanitizers else '-O2','-Wall','-Wextra','-Werror',
             '-I',str(HEADER.parent),str(cpp),'-o',str(exe)]
        if opts.sanitizers:cmd+=['-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']
        built=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        (out/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('compile failed: '+built.stderr[-4000:])
        env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'}
        run=subprocess.run([str(exe)],capture_output=True,text=True,env=env,timeout=30)
        (out/'run.log').write_text(run.stdout+run.stderr)
        if run.returncode or 'PASS' not in run.stdout:raise RuntimeError('host bridge mismatch: '+run.stderr)
        result.update(status='PASS_SOURCE_EXTRACTED_HOST_ONLY',output=run.stdout.strip())
    except Exception as exc:
        result.update(status='FAIL_SOURCE_EXTRACTED_HOST_ONLY',error=str(exc))
        save();print(str(exc),file=sys.stderr);return 1
    save();print(json.dumps(result,ensure_ascii=False));return 0

if __name__=='__main__':sys.exit(main())
