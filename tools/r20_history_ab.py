#!/usr/bin/env python3
"""Three-way mirrored CPU-only comparison of the unchanged R19 and R20 stacks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import zipfile

KIT = Path(__file__).resolve().parents[1]
NAMES = ("r19_all_off", "r19_cpu_stack", "r20_cpu_stack")
ALL_NAMES = ("upstream", *NAMES, "r19_unstaged_prefetch")
STACK = ("HASH_SCRATCH", "PREFETCH_UNIQUE", "ENGRAM_STAGE", "ENGRAM_MODEL_CACHE", "ENGRAM_ASYNC")


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("fixture", "upstream-bin", "historical-bin", "candidate-bin", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--core", type=int, required=True)
    args = p.parse_args()
    if args.core not in os.sched_getaffinity(0):
        p.error("CPU not in this process's permitted affinity")
    args.out = args.out.resolve()
    try:
        args.out.mkdir(mode=0o700)
    except FileExistsError:
        p.error("--out must be new")
    bins = [x.resolve() for x in (args.upstream_bin, args.historical_bin, args.historical_bin, args.candidate_bin, args.historical_bin)]
    common = None
    libraries = {}
    for directory in dict.fromkeys(bins):
        cache = {}
        for line in (directory.parent/"CMakeCache.txt").read_text().splitlines():
            if re.match(r"(CMAKE_(BUILD_TYPE|C_COMPILER|CXX_COMPILER|CXX_FLAGS_RELEASE|C_FLAGS_RELEASE)|GGML_(NATIVE|CUDA|AVX.*|FMA|F16C|BMI2)):", line):
                k,v = line.split("=",1);cache[k]=v
        if common is not None and cache != common:
            raise ValueError("unequal native build options")
        common = cache
        libraries[directory.parent.name] = {x:sha(directory/x) for x in ("libllama.so","libggml-cpu.so","libggml.so","libggml-base.so")}
    driver = (args.fixture/"decode").resolve()
    model = (args.fixture/"synthetic.gguf").resolve()
    off = next(x["environment"] for x in json.loads((KIT/"profiles/experiments.json").read_text())["profiles"] if x["name"]=="candidate_all_off")
    flags = {**off, **{"LLAMA_EXPERIMENTAL_"+k:"1" for k in STACK}}
    env_modes = [off, off, flags, {**flags, "LLAMA_EXPERIMENTAL_CPU_IQ3_SIGNMASK":"1"},
                 {**off,"LLAMA_EXPERIMENTAL_HASH_SCRATCH":"1","LLAMA_EXPERIMENTAL_PREFETCH_UNIQUE":"1"}]
    report = dict(scope="synthetic CPU-only untrained 6-layer DS4.1, 512 context, 3 requests/run, n_gpu_layers=0; no target inference",
                  status="RUNNING", core=args.core, model_sha256=sha(model), model_bytes=model.stat().st_size,
                  driver_sha256=sha(driver), cmake=common, libraries=libraries, modes=dict(zip(ALL_NAMES,env_modes)),
                  warmups="one full unmeasured run of each role; fresh processes per run; OS cache warm",
                  tests=[], summary={})
    ref = None
    base_env = {k:v for k,v in os.environ.items() if not k.startswith(("GGML_","LLAMA_"))}
    def run(arm, block):
        nonlocal ref
        name = ALL_NAMES[arm]
        key = ("warm" if block is None else "b"+str(block))+"_"+name
        out,err,logits = (args.out/(key+"."+suf) for suf in ("stdout","stderr","f32"))
        command = ["taskset", "-c", str(args.core), str(driver), str(model), str(logits), "256"]
        with out.open("xb") as a, err.open("xb") as b:
            proc = subprocess.run(command, env=dict(base_env, LD_LIBRARY_PATH=str(bins[arm]), **env_modes[arm]),
                                  stdout=a, stderr=b, timeout=240)
        stderr=err.read_text()
        ids=[[int(x) for x in line.split()] for line in out.read_text().splitlines()]
        row=dict(role=name,block=block,exit_code=proc.returncode,stdout=out.name,stderr=err.name)
        report["tests"].append(row)
        if proc.returncode or len(ids)!=3 or any(len(v)!=257 for v in ids) or ids[0]!=ids[2] or logits.stat().st_size!=(95+3*256)*128*4:
            raise ValueError(f"{key} invalid output or process failure")
        row.update(timings=json.loads(stderr.split("R20_SYNTHETIC_TIMINGS ")[1].splitlines()[0]),
                   ids_sha256=sha(out),logits_sha256=sha(logits),logit_f32_count=logits.stat().st_size//4,
                   stage_graph_seen="engram_stage-" in stderr, iq3_entered="experimental_cpu_iq3_signmask: entered" in stderr)
        if row["stage_graph_seen"] != (arm in (2,3)) or row["iq3_entered"] != (arm==3):
            raise ValueError(f"{key} unexpected runtime branch: {row}")
        parity=(row["ids_sha256"],row["logits_sha256"])
        if ref is None: ref=parity
        if parity!=ref:
            raise ValueError(f"{key} bit mismatch against pinned upstream")
    try:
        for arm in range(len(ALL_NAMES)):
            run(arm,None)
        for block,order in enumerate(("123","321","231","132","312","213")):
            for arm in order:
                run(int(arm),block)
        for name in NAMES:
            rs=[r["timings"] for r in report["tests"] if r["role"]==name and r["block"] is not None]
            metrics=report["summary"][name]={}
            for mode in ("pp","tg"):
                samples=sorted(1e9*r[mode+"_tokens"]/r[mode+"_ns"] for r in rs)
                metrics[mode]=dict(n=len(samples),median_tok_s=statistics.median(samples),p95_tok_s=samples[math.ceil(.95*len(samples))-1],
                                   sample_variance=statistics.variance(samples))
        report["status"]="PASS"
    finally:
        with (args.out/"result.json").open("w") as f: json.dump(report,f,indent=2)
        with zipfile.ZipFile(args.out/"stdout-stderr.zip","w",zipfile.ZIP_DEFLATED) as logs:
            for path in sorted(args.out.iterdir()):
                if path.suffix in (".stdout",".stderr"): logs.write(path,path.name)
    print(json.dumps(report["summary"]))


if __name__=="__main__":
    main()
