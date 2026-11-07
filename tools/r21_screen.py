#!/usr/bin/env python3
"""Portable replay of the local R21 screening protocol; not a quality acceptance test."""
import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import r21_server as server
import r20_bench as bench
from overlay import sha256_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='First GGUF shard')
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, default=server.KIT/'benchmarks/prompts/moe-russian.txt')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--predict', type=int, default=256)
    p.add_argument('--timeout', type=int, default=7200)
    v = p.parse_args(argv)
    if not 1 <= v.repeats <= 100 or not 1 <= v.predict <= 4096 or v.timeout < 1:
        p.error('Invalid repeats, predict or timeout')
    cfg = json.loads(v.profile.read_text())
    if cfg['parallel'] != 1 or cfg['spec_type'] != 'none':
        p.error('This replay implements only single-slot, non-speculative decoding')
    a = SimpleNamespace(model=v.model.resolve(), out=v.out.resolve(),
        ctx=cfg['ctx'], predict=v.predict, ngl=cfg['ngl'], moe_cache_mib=cfg['cache_mib'],
        threads=cfg['threads'], batch=cfg['batch'], ubatch=cfg['ubatch'],
        cache_type_k=cfg['kv'], cache_type_v=cfg['kv'], cpu_mask=cfg.get('cpu_mask'),
        cpu_strict=True, timeout=v.timeout, mtp=False, draft_model=None, draft_max=4)
    binary = v.binary.resolve()
    if not binary.is_file() or not a.model.is_file():
        p.error('Binary or model missing')
    # Reuse the existing shard validation without its build-pair requirements.
    import re
    shard = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', a.model.name)
    models = [a.model]
    if shard:
        if shard[1] != '00001' or int(shard[2]) < 1:
            p.error('Specify the first GGUF shard')
        models = [a.model.with_name(a.model.name[:shard.start()] + f'-{i:05d}-of-{int(shard[2]):05d}.gguf')
                  for i in range(1, int(shard[2])+1)]
        if any(not x.is_file() for x in models):
            p.error('Incomplete GGUF shard set')
    prompt = v.prompt_file.read_text()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    server.new_output(p, a.out)
    report = {'status': 'RUNNING', 'scope': 'SCREENING_NOT_ACCEPTANCE', 'profile': cfg, 'runs': [],
              'binary_sha256': sha256_file(binary),
              'library_sha256': {x.name: sha256_file(x) for x in sorted(binary.parent.glob('*.so*')) if x.is_file()},
              'gguf_inventory': [{'name': x.name, 'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns} for x in models],
              'inventory_is_not_a_weight_content_hash': True,
              'prompt_file_sha256': sha256_file(v.prompt_file)}
    code = 1
    try:
        with server.server(binary, 'screen', cfg['environment'], a) as (port, launch):
            report['launch'] = launch
            formatted = server.request(port, 'apply-template', {'messages': [{'role':'user', 'content':prompt}],
                'chat_template_kwargs': {'enable_thinking': False}}, a.timeout)['prompt']
            ids = server.request(port, 'tokenize', {'content':formatted, 'add_special':False, 'parse_special':True}, a.timeout)['tokens']
            if len(ids) + a.predict >= a.ctx:
                raise ValueError('Prompt and generation exceed the configured context')
            report['prompt_ids'] = ids
            report['warmup'] = bench.warmup(port, a)
            for index in range(v.repeats):
                body = bench.payload(ids, a.predict)
                body['ignore_eos'] = False
                start = time.perf_counter()
                response = server.request(port, 'completion', body, a.timeout)
                elapsed = time.perf_counter() - start
                tokens = response['tokens']
                if not tokens or len(tokens) > a.predict or any(type(x) is not int or x < 0 for x in tokens):
                    raise ValueError('Invalid generated token IDs')
                bench.validate_completion(response, len(ids), len(tokens))
                if len(tokens) < a.predict and response['stop_type'] != 'eos':
                    raise ValueError('Unexpected short completion')
                report['runs'].append({'tokens':tokens, 'timings':response['timings'],
                    'native_eog':response['stop_type']=='eos', 'request_s':elapsed})
                bench.atomic_write(a.out/'SPEED.json', report)
                print(index, response['timings']['predicted_per_second'], flush=True)
        report['status'] = 'SCREENING_COMPLETE'
        code = 0
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        report.update(status='FAIL', error=str(exc) or type(exc).__name__)
    finally:
        bench.atomic_write(a.out/'SPEED.json', report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
