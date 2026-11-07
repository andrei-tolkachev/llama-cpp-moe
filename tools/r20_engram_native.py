#!/usr/bin/env python3
"""Linux: native CPU graph parity on a small deterministic, untrained DS4.1 GGUF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import time
import statistics
import math
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "control-bin", "candidate-bin", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--bench", action="store_true", help="include six balanced 3-arm native PP/TG blocks")
    parser.add_argument("--tg", type=int, default=256, help="single-token decode steps per request")
    parser.add_argument("--order-seed", type=int, help="shuffle the six complete within-block orders")
    args = parser.parse_args()
    if not 1 <= args.tg <= 400:
        parser.error("--tg must be 1..400")
    src, control, candidate, out = (x.resolve() for x in
                                    (args.source, args.control_bin, args.candidate_bin, args.out))
    fixtures = Path(__file__).resolve().parents[1] / "tests_cpp"
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--out must be new")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
    report = dict(scope="synthetic untrained CPU DeepSeek 4.1; mmap/lazy Q6_K Engram, 6 layers, "
                        "4 IQ3_XXS experts per layer (2 active); 128 vocabulary items, 512-token KV context",
                  os=platform.platform(), affinity=sorted(os.sched_getaffinity(0)), order_seed=args.order_seed,
                  tg_single_steps_per_request=args.tg, source_fixture_sha256=sha(src / "tests/test-llama-archs.cpp"),
                  compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                  libs={k: sha(lib / "libllama.so") for k, lib in [("control", control), ("candidate", candidate)]},
                  commands=[], tests=[])
    def execute(name, cmd, lib, flags=None):
        env_run = dict(env, LD_LIBRARY_PATH=str(lib), **(flags or {}))
        cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.monotonic()
        with (out / (name + ".stdout")).open("w") as a, (out / (name + ".stderr")).open("w") as b:
            proc = subprocess.run(cmd, env=env_run, stdout=a, stderr=b, timeout=180)
        wall = time.monotonic() - t0
        cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        report["commands"].append(dict(whole_process_wall_s=wall,
                                       whole_process_cpu_s=(cpu1.ru_utime - cpu0.ru_utime + cpu1.ru_stime - cpu0.ru_stime),
                                       name=name, cmd=list(map(str, cmd)), library_path=str(lib),
                                       flags=flags, status=proc.returncode))
        if proc.returncode:
            raise RuntimeError(f"{name} failed; see {(out / (name + '.stderr'))}")
    try:
        for label in ["make", "decode"]:
            exe = out / label
            cmd = ["g++", "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections"]
            cmd.extend("-I" + str(src / x) for x in ["tests", "include", "common", "src", "ggml/include", "vendor"])
            execute("compile-" + label, [*cmd, fixtures / f"r20_ds41_fixture_{label}.cpp", "-L" + str(control),
                    "-Wl,-rpath-link," + str(control), "-lllama", "-lggml", "-lggml-base", "-o", exe], control)
        model = out / "synthetic.gguf"
        execute("make-model", [out / "make", model], control)
        report["model"] = {"sha256": sha(model), "file_bytes": model.stat().st_size}
        reference = None
        modes = [
                ("upstream", control, (0, 0, 0)), ("off", candidate, (0, 0, 0)),
                ("local", candidate, (1, 0, 0)), ("shared", candidate, (1, 1, 0)),
                ("async", candidate, (1, 1, 1)), ("signmask", candidate, (0, 0, 0))]
        schedule = [(None, *mode) for mode in modes]
        if args.bench:
            # Each of three variants occupies each phase of a block twice.
            orders = ["012", "210", "120", "021", "201", "102"]
            if args.order_seed is not None:
                random.Random(args.order_seed).shuffle(orders)
            for block, order in enumerate(orders):
                for index in order:
                    schedule.append((block, *modes[[0, 1, 5][int(index)]]))
        for block, role, lib, switches in schedule:
            label = role if block is None else f"b{block}_{role}"
            flags = {"LLAMA_EXPERIMENTAL_ENGRAM_" + k: str(v)
                     for k, v in zip(("STAGE", "MODEL_CACHE", "ASYNC"), switches)}
            flags["LLAMA_EXPERIMENTAL_ENGRAM_TRACE"] = "1" if block is None else "0"
            flags["LLAMA_EXPERIMENTAL_CPU_IQ3_SIGNMASK"] = "1" if role == "signmask" else "0"
            logits = out / (label + ".f32")
            execute(label, [out / "decode", model, logits, str(args.tg)], lib, flags)
            stderr = (out / (label + ".stderr")).read_text()
            signmask_entered = "experimental_cpu_iq3_signmask: entered" in stderr
            if (role == "signmask") != signmask_entered:
                raise RuntimeError(f"{label} dispatched signmask kernel = {signmask_entered}")
            trace = [json.loads(line.partition("R20_ENGRAM_JSON ")[2]) for line in
                     (out / (label + ".stderr")).read_text().splitlines() if "R20_ENGRAM_JSON " in line]
            ids = [[int(x) for x in line.split()] for line in (out / (label + ".stdout")).read_text().splitlines()]
            if len(ids) != 3 or any(len(x) != args.tg + 1 for x in ids) or ids[0] != ids[2]:
                raise RuntimeError(f"{label} repetition/count mismatch")
            if logits.stat().st_size != (41 + 13 + 41 + args.tg * 3) * 128 * 4:
                raise RuntimeError(f"{label} logit count mismatch")
            current = (sha(logits), ids)
            if reference is None:
                reference = current
            if current != reference:
                raise RuntimeError(f"{label} disagrees with control logit bits or greedy IDs")
            count = sum(x["rows_completed"] for x in trace)
            if switches[0] and (not count or any(x["read_failures"] for x in trace)):
                raise RuntimeError(f"{label} Engram staging missing or failed")
            timings = next(json.loads(line.partition("R20_SYNTHETIC_TIMINGS ")[2])
                           for line in (out / (label + ".stderr")).read_text().splitlines()
                           if line.startswith("R20_SYNTHETIC_TIMINGS "))
            report["tests"].append(dict(label=label, role=role, block=block, timings=timings, signmask_entered=signmask_entered, logits_sha256=current[0], logits_f32=logits.stat().st_size // 4,
                                        greedy_ids=ids, row_counters=trace))
        if args.bench:
            report["summary"] = {}
            for role in ("upstream", "off", "signmask"):
                runs = [r["timings"] for r in report["tests"] if r["block"] is not None and r["role"] == role]
                stats = {}
                for kind in ("pp", "tg"):
                    samples = [1e9 * r[kind + "_tokens"] / r[kind + "_ns"] for r in runs]
                    stats[kind] = dict(n=len(samples), median_tok_s=statistics.median(samples),
                                       p95_tok_s=sorted(samples)[math.ceil(len(samples)*.95)-1],
                                       sample_variance=statistics.variance(samples))
                report["summary"][role] = stats
        report["status"] = "PASS"
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: " + str(out / "result.json"))


if __name__ == "__main__":
    main()
