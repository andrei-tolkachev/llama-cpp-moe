#!/usr/bin/env python3
"""P10 source-regression bridge: reconstitute unchanged P04/P09B patched hunks.

This is a host-only deferred-backend test of the EXACT earlier patch text,
compiled against the P10 vendor module. Does not fetch or compile full upstream.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

from test_moe_pipeline_bridge import run_fixture

ROOT=Path(__file__).resolve().parents[1]


def after_hunks(patch: Path) -> str:
    lines = []
    in_hunk = False
    for line in patch.read_text().splitlines():
        if line.startswith('@@ '):
            in_hunk = True
        elif line.startswith('diff --git '):
            in_hunk = False
        elif in_hunk and line and line[0] in ('+', ' '):
            lines.append(line[1:])
    return '\n'.join(lines)+'\n'


def extract(data: str, first: str, last: str):
    if data.count(first)!=1 or data.count(last)!=1:
        raise ValueError('required source anchor absent or repeated')
    i=data.index(first);j=data.index(last,i+len(first))
    if j<=i:raise ValueError('required source anchor order invalid')
    return data[i:j]


def main() -> int:
    try:
        patch4=ROOT/'patches/0004-moe-upload-staging.patch'
        patch9=ROOT/'patches/0008-moe-async-upload-pipeline-integration.patch'
        manifest=json.loads((ROOT/'manifest.json').read_text())
        hashes={p['file']:p['sha256'] for p in manifest['patches']}
        for patch in (patch4,patch9):
            if hashlib.sha256(patch.read_bytes()).hexdigest()!=hashes[patch.name]:
                raise ValueError('prior patch byte integrity failed')
        p04=extract(after_hunks(patch4),'struct moe_cache_upload_stats {',
                    '// gate, up, down or gate_up, down')
        p09=extract(after_hunks(patch9),
                    '        // P09B: the producer only prepares disjoint host upload slices.',
                    '        stats & st = ids.size()')
        with tempfile.TemporaryDirectory(prefix='llama-exp-p10-source-') as td:
            td=Path(td);candidate=td/'mock_candidate';src=candidate/'src';src.mkdir(parents=True)
            hdr=src/'llama-experimental';hdr.mkdir()
            for name in ('upload_types.hpp','upload_worker.hpp','upload_pipeline.hpp','upload_plan.hpp'):
                expected=ROOT/'modules/moe'/name
                records=[x for x in manifest['additional_files'] if x['source']=='modules/moe/'+name]
                if len(records)!=1 or hashlib.sha256(expected.read_bytes()).hexdigest()!=records[0]['sha256']:
                    raise ValueError('vendor file not manifest verified: '+name)
                shutil.copyfile(expected,hdr/name)
            output=td/'test_result'
            output.mkdir()
            os.environ['LLAMA_EXPERIMENTAL_MOE_PERSISTENT']='1'
            rc=run_fixture(candidate,output,'c++',True,p04,p09,
                           hashlib.sha256((p04+p09).encode()).hexdigest(),
                           'P04/P09B patch hunks with persistent worker')
            if rc:
                return rc
            report=json.loads((td/'test_result/result.json').read_text())
            if report['status']!='PASS' or not report['sanitizers']:
                raise ValueError('source bridge sanitizer failure')
            print(json.dumps({'status':'PASS','scope':'P04+P09B unchanged patch hunks + P10 persistent vendor',
                'mock_backend':report['status'],'host_sanitizers':True,'steps':len(report['steps']),
                'actual_full_llama_cpp_compilation':False}))
        return 0
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status':'ERROR','reason':str(exc)}))
        return 2

if __name__=='__main__':sys.exit(main())
