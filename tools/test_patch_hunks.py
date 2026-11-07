#!/usr/bin/env python3
"""Round-trip P08 patch against a synthetic hunk-backed source fixture.

This deliberately does NOT claim application to the full pinned upstream source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

def hash_bytes(x):return hashlib.sha256(x).hexdigest()

def exec_git(root,*args):
    r=subprocess.run(['git','-c','core.autocrlf=false',*args],cwd=root,
                     capture_output=True,text=True,timeout=40)
    if r.returncode:raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.source_root.resolve(strict=True)
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(root) or not output.parent.is_dir():
        parser.error('--output must be new outside source tree')
    manifest=json.loads((root/'manifest.json').read_text())
    patch_entry=next(p for p in manifest['patches'] if p['id']=='P08-deepseek41-engram-graph-stage')
    assert patch_entry['id']=='P08-deepseek41-engram-graph-stage'
    patch_path=root/'patches'/patch_entry['file']
    raw=patch_path.read_bytes()
    assert hash_bytes(raw)==patch_entry['sha256']
    lines=raw.decode().splitlines(keepends=True)
    assert lines[0].startswith('diff --git a/src/models/deepseek41.cpp')
    hunk_re=re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
    old_lines=['// untouched fixture position %04d\n'%i for i in range(1,1100)]
    hunks=0
    for j,line in enumerate(lines):
        m=hunk_re.match(line)
        if not m:continue
        hunks+=1
        old_start,old_count,new_start,new_count=map(int,m.groups())
        pos=old_start-1
        actual_old=actual_new=0
        for content in lines[j+1:]:
            if content.startswith('@@'):break
            if content.startswith('diff --git'):break
            if content.startswith(('---','+++')):break
            sign=content[:1]
            if sign in (' ','-'):
                if not old_lines[pos].startswith('// untouched') and old_lines[pos]!=content[1:]:
                    raise RuntimeError('Overlapping hunk conflict')
                old_lines[pos]=content[1:]; pos+=1;actual_old+=1
            if sign in (' ','+'):actual_new+=1
        if actual_old!=old_count or actual_new!=new_count:
            raise RuntimeError('Malformed unified hunk record')
    output.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='llama-p08-hunks-') as tmp:
        git_root=Path(tmp)
        (git_root/'src/models').mkdir(parents=True)
        source=git_root/'src/models/deepseek41.cpp'
        source.write_text(''.join(old_lines))
        original=source.read_bytes()
        exec_git(git_root,'init','-q')
        exec_git(git_root,'config','user.email','fixture@example.invalid')
        exec_git(git_root,'config','user.name','Synthetic Fixture')
        exec_git(git_root,'add','--all')
        exec_git(git_root,'commit','-qm','synthetic')
        exec_git(git_root,'apply','--check','--whitespace=error-all',str(patch_path))
        exec_git(git_root,'apply','--whitespace=error-all',str(patch_path))
        modified=source.read_bytes()
        if original==modified:raise RuntimeError('Patch did not modify fixture')
        exec_git(git_root,'apply','--reverse','--check',str(patch_path))
        exec_git(git_root,'apply','--reverse',str(patch_path))
        if source.read_bytes()!=original:raise RuntimeError('Reverse patch is not byte-exact')
    report={'status':'PASS','test':'P08 synthetic hunk source only','hunks':hunks,
            'source_sha256':hash_bytes(original),'modified_sha256':hash_bytes(modified),
            'patch_sha256':hash_bytes(raw),'roundtrip_exact':True,
            'pinned_full_source_applied':False,'native_engine_compiled':False}
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','hunks':hunks,'output':str(output/'result.json')}))

if __name__=='__main__':main()
