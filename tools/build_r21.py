#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import subprocess
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--arch', default='120a')
p.add_argument('--cuda-compiler', type=Path)
p.add_argument('--ui-assets', type=Path)
p.add_argument('--production-tools', action='store_true')
p.add_argument('--jobs', type=int, default=8)
a = p.parse_args()
source, build = a.source.resolve(), a.build.resolve()
if build.exists():
    p.error('Build destination must not exist; preserve previous builds')
if not (source / 'CMakeLists.txt').is_file() or not 1 <= a.jobs <= 32:
    p.error('Invalid source or jobs')
build.mkdir(parents=True)
if a.ui_assets:
    assert (a.ui_assets/'index.html').is_file()
    shutil.copytree(a.ui_assets, build/'tools/ui/dist')
configure = ['cmake', '-S', str(source), '-B', str(build),
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
             '-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON', '-DLLAMA_BUILD_COMMON=ON',
             '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_TESTS=ON',
             '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_UI=OFF',
             '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_CONNECT=OFF', '-DLLAMA_LLGUIDANCE=OFF',
             '-DLLAMA_USE_SYSTEM_GGML=OFF', '-DGGML_CPU=ON', '-DGGML_METAL=OFF',
             '-DGGML_CCACHE=OFF', '-DGGML_CPU_KLEIDIAI=OFF', '-DGGML_CUDA=ON',
             '-DGGML_OPENMP_FETCH=OFF', '-DGGML_CUDA_CCCL_VERSION=', '-DGGML_CUDA_NCCL=OFF',
             '-DLLAMA_BUILD_MTMD=OFF', '-DCMAKE_C_COMPILER_LAUNCHER=',
             '-DCMAKE_CXX_COMPILER_LAUNCHER=', '-DCMAKE_CUDA_COMPILER_LAUNCHER=',
             '-DGGML_NATIVE=ON', '-DGGML_OPENMP=ON', '-DCMAKE_CUDA_ARCHITECTURES=' + a.arch]
if a.production_tools:
    configure += ['-DLLAMA_BUILD_EXAMPLES=ON', '-DLLAMA_BUILD_APP=ON', '-DLLAMA_BUILD_TESTS=OFF']
if a.cuda_compiler:
    configure.append('-DCMAKE_CUDA_COMPILER=' + str(a.cuda_compiler.resolve()))
compile = ['cmake', '--build', str(build), '-j', str(a.jobs), '--target',
           'llama-server', 'llama-perplexity', 'test-backend-ops']
if a.production_tools:
    compile = ['cmake', '--build', str(build), '-j', str(a.jobs)]
record = {'status': 'BUILDING', 'source': str(source), 'commands': [configure, compile],
          'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt = source / '.llama-overlay-receipt.json'
if receipt.is_file():
    record['overlay_receipt'] = json.loads(receipt.read_text())
(build / 'BUILD.json').write_text(json.dumps(record, indent=2))
try:
    with (build / 'build.log').open('w') as log:
        for cmd in record['commands']:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    artifacts = {x.resolve() for x in (build / 'bin').iterdir() if x.is_file()}
    record['binary_sha256'] = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(artifacts)}
    record['status'] = 'BUILT'
except subprocess.CalledProcessError as e:
    record.update(status='BUILD_FAILED', exit_code=e.returncode)
    raise
finally:
    record['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (build / 'BUILD.json').write_text(json.dumps(record, indent=2))
print(record['status'], build)
