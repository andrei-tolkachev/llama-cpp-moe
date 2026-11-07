#!/usr/bin/env python3
"""Within-host crossover benchmark. Use a fresh directory for each prompt/configuration."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import time
import urllib.request

import p17_smoke as smoke


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


HISTORICAL_R19_MANIFEST = "79c9db6cab118b5fc6931b4d4b431c15fdc7ac27a7ebc206b51fd89f01bd4387"


def provenance(args, builds=None, expected_manifest=None):
    builds = builds if builds is not None else args.builds
    path = builds / "build_record.json"
    rec = json.loads(path.read_text())
    manifest = smoke.KIT / "manifest.json"
    ver = rec["plan"]["verification"]
    if (rec.get("status") != "BUILD_ONLY_KNOWN_BASE" or rec.get("exit_code") != 0
            or ver["base_commit"] != json.loads(manifest.read_text())["upstream"]["base_commit"]
            or ver["manifest_sha256"] != (expected_manifest or file_hash(manifest))):
        raise ValueError("Wrong or incomplete build receipt. Rebuild from this kit.")
    if args.ngl and (rec["plan"].get("cuda_requested") is not True
                     or rec["plan"].get("cuda_arch_requested") != "120"):
        raise ValueError("Expected sm_120 CUDA pair in build receipt")
    hashes = {}
    for role in ("baseline", "candidate"):
        build = builds / role
        expected = {a["path"]: a["sha256"] for a in rec["builds"][role]["artifacts"]}
        files = {p for p in (build / "bin").glob("*.so*") if p.is_file()}
        files.add(build / "bin/llama-server")
        for p in sorted(files):
            digest = hashes.get(str(p.resolve()))
            if digest is None:
                digest = hashes[str(p.resolve())] = file_hash(p)
            if digest != expected.get(p.relative_to(build).as_posix()):
                raise ValueError(f"Unrecorded or changed executable/library: {p}")
    return {"build_record_sha256": file_hash(path), "base_commit": ver["base_commit"],
            "manifest_sha256": ver["manifest_sha256"], "checked_artifacts": hashes}


def model_inventory(model):
    """Fast identity check of every shard; full hashing hundreds of GiB warms the disk cache."""
    paths = [model]
    shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", model.name)
    if shard:
        paths = [model.with_name(f"{model.name[:shard.start()]}-{i:05}-of-{shard[2]}.gguf")
                 for i in range(1, int(shard[2]) + 1)]
    inventory = {}
    for path in paths:
        st = path.stat()
        inventory[str(path)] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
    return inventory


def bundle_flags(profiles, include_r20=False):
    """Explicit multi-flag DS4.1 hypothesis. Timing flags never enable trace or a drafter."""
    env = profiles["candidate_all_off"].copy()
    prefix = "LLAMA_EXPERIMENTAL_"
    flags = ["MOE_PLAN_REUSE", "DSV4_HC_POST_LAYERS", "HASH_SCRATCH", "PREFETCH_UNIQUE",
             "MOE_PIPELINE", "MOE_PERSISTENT", "MOE_SHARED_PLAN", "ENGRAM_STAGE",
             "ENGRAM_MODEL_CACHE", "ENGRAM_ASYNC", "MOE_SCHEDULER", "MOE_EVENT_FENCE",
             "DSV4_HC_POST_REUSE", "CUDA_MMQ_TILE_CACHE"]
    if include_r20:
        flags.extend(["CPU_IQ3_SIGNMASK", "MOE_BYPASS_NO_PACK", "MOE_DEMAND_READAHEAD"])
    if not all(prefix + f in env for f in flags):
        raise ValueError("Full bundle flags are missing from this kit")
    env.update({prefix + f: "1" for f in flags})
    env[prefix + "MOE_STAGING_MIB"] = "32"
    return env


def historical_flags(env):
    r20_only = {"LLAMA_EXPERIMENTAL_" + name for name in
                ("CPU_IQ3_SIGNMASK", "MOE_BYPASS_NO_PACK", "MOE_DEMAND_READAHEAD", "ENGRAM_TRACE")}
    return {k: v for k, v in env.items() if k not in r20_only}


def crossover(labels, repeats, seed):
    """Randomized complete blocks; adjacent pairs have exactly reversed orders."""
    rng = random.Random(seed)
    order = labels.copy()
    blocks = []
    for i in range(repeats):
        if i % 2 == 0:
            rng.shuffle(order)
        else:
            order.reverse()
        blocks.append(order.copy())
    return blocks


def tokenized(port, text, timeout):
    result = smoke.request(port, "tokenize", {"content": text, "add_special": True}, timeout)["tokens"]
    if not isinstance(result, list) or not result or any(type(t) is not int or t < 0 for t in result):
        raise ValueError("Invalid tokenizer response")
    return result


def prompt_ids(port, args):
    n = args.fill_tokens
    text = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else smoke.PROMPT
    if n and not args.prompt_file:
        # Tokenize a complete text; truncation is applied to actual tokenizer IDs.
        text += "".join(f"\nRecord {i}. Copper and water. Запись {i}: ветер у реки."
                        for i in range(n // 6 + 16))
    ids = tokenized(port, text, args.timeout)
    if n and len(ids) < n:
        raise ValueError(f"Input produced {len(ids)} tokens, need {n}. Supply a longer --prompt-file")
    ids = ids[:n] if n else ids
    if len(ids) + args.predict >= args.ctx:
        raise ValueError(f"Tokenized prompt ({len(ids)}) + output ({args.predict}) must be < {args.ctx}")
    return ids


def payload(ids, predict, stream=False):
    return {"prompt": ids, "temperature": 0, "seed": 42, "n_predict": predict, "return_tokens": True,
            "ignore_eos": True, "cache_prompt": False, "stream": stream}


def validate_completion(response, n_input, n_output):
    timings = response["timings"]
    if (response["truncated"] is not False or response["tokens_evaluated"] != n_input
            or timings.get("cache_n", 0) != 0 or timings["prompt_n"] != n_input or response["tokens_predicted"] != n_output
            or timings["predicted_n"] != n_output):
        raise ValueError("Cached, truncated, or incomplete request: "
                         f"input={n_input}, output={n_output}, timings={timings}, "
                         f"truncated={response['truncated']}")
    for key in ("prompt_per_second", "predicted_per_second", "prompt_ms", "predicted_ms"):
        if not math.isfinite(timings[key]) or timings[key] <= 0:
            raise ValueError(f"Invalid server timing {key}: {timings[key]}")


def exact_completion(port, prompt, predict, timeout):
    start = time.perf_counter()
    res = smoke.request(port, "completion", payload(prompt, predict), timeout)
    duration = time.perf_counter() - start
    validate_completion(res, len(prompt), predict)
    ids = res["tokens"]
    if not isinstance(ids, list) or len(ids) != predict or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("Server did not provide all generated token IDs")
    return {"tokens": ids, "timings": res["timings"], "request_s": duration}


def completion_stream(port, prompt, predict, timeout, on_first=None):
    """Measure first client-observed token event; SSE can omit incomplete UTF-8 token IDs."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 json.dumps(payload(prompt, predict, True)).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    count = 0
    final = None
    with smoke.API.open(req, timeout=timeout) as resp:
        if "text/event-stream" not in resp.headers.get("Content-Type", ""):
            raise ValueError("Expected a streaming SSE response")
        for line in resp:
            arrived = time.perf_counter() - start
            if arrived > timeout:
                raise TimeoutError("Streaming completion deadline exceeded")
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            row = json.loads(data)
            if "error" in row:
                raise ValueError(str(row["error"]))
            if row.get("stop") is True:
                final = row
                break
            if not row.get("prompt_progress") and row.get("tokens"):
                if first is None:
                    first = arrived
                    if on_first is not None:
                        on_first()
                count += len(row["tokens"])
    if first is None or final is None:
        raise ValueError("No client-observed token event or final SSE result")
    validate_completion(final, len(prompt), predict)
    return {"ttft_s": first, "emitted_ids_count": count, "tokens_predicted": predict,
            "timings": final["timings"]}


def warmup(port, args):
    prompt = tokenized(port, "Warm up. Count from one to sixteen.", args.timeout)
    return exact_completion(port, prompt, 16, args.timeout)


def summarize(values):
    return {"n": len(values), "median": statistics.median(values),
            "p95": sorted(values)[math.ceil(.95 * len(values)) - 1],
            "variance": statistics.variance(values) if len(values) > 1 else None}


def summarize_runs(runs):
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        group = groups[r["variant"]]
        for k in ("startup_to_health_s", "ttft_s", "request_s"):
            group[k].append(r[k])
        for k in ("prompt_per_second", "predicted_per_second", "prompt_ms", "predicted_ms"):
            group[k].append(r["timings"][k])
    return {label: {k: summarize(vals) for k, vals in g.items()} for label, g in groups.items()}


def paired_gains(runs, pairs, repeats):
    blocks = {(r["block"], r["variant"]): r for r in runs}
    result = []
    for control, experiment in pairs:
        entry = {"control": control, "experiment": experiment}
        for key in ("prompt_per_second", "predicted_per_second"):
            delta = []
            for i in range(repeats):
                c = blocks[i, control]["timings"][key]
                e = blocks[i, experiment]["timings"][key]
                delta.append((e / c - 1) * 100)
            entry[key] = {"paired_pct_by_block": delta, **summarize(delta)}
        result.append(entry)
    return result


def atomic_write(path, data):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def compare_ids(reference, row):
    if row["prompt_ids_sha256"] != reference["prompt_ids_sha256"]:
        raise ValueError("Tokenized input differs between launches")
    if row["offload_layers"] != reference["offload_layers"]:
        raise ValueError("Effective number of offloaded layers changed")
    if row["tokens"] != reference["tokens"]:
        mismatch = next(i for i, (a, b) in enumerate(zip(row["tokens"], reference["tokens"])) if a != b)
        raise ValueError(f"Greedy ID differs from {reference['label']} at index {mismatch}: "
                         f"{row['tokens'][mismatch]} vs {reference['tokens'][mismatch]}")


def high_water_kib(pid):
    try:
        if os.readlink("/proc/self") != str(os.getpid()):
            return None
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def main(argv=None):
    profiles = {p["name"]: p["environment"] for p in
                json.loads((smoke.KIT / "profiles/experiments.json").read_text())["profiles"]}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--builds", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", choices=sorted(profiles), action="append",
                   help="One flag relative to --control. May repeat. Default: planner_only.")
    p.add_argument("--bundle", choices=["r19-compatible"],
                   help="Predefined full compatible DeepSeek configuration; reports flag deltas")
    p.add_argument("--r19-builds", type=Path,
                   help="Historical R19 builds directory with the original build_pair receipt (bundle only)")
    p.add_argument("--control", choices=sorted(profiles), default="candidate_all_off")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--predict", type=int, default=256)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ngl", type=int, default=18)
    p.add_argument("--moe-cache-mib", type=int, default=12288)
    p.add_argument("--order-seed", type=int, default=20261103)
    p.add_argument("--timeout", type=int, default=3600, help="Seconds per model load or request")
    p.add_argument("--prompt-file", type=Path, help="UTF-8 text; tokenize once per launch")
    length = p.add_mutually_exclusive_group()
    length.add_argument("--fill-tokens", type=int, default=0, help="Use exactly N real tokenizer IDs")
    length.add_argument("--near-full", action="store_true", help="Use ctx - predict - 1 input tokens")
    args = p.parse_args(argv)
    args.mtp = False
    if args.near_full:
        args.fill_tokens = args.ctx - args.predict - 1
    if args.repeats < 5 or not 256 <= args.predict <= 512 or args.ctx < 131072:
        p.error("Acceptance requires >=5 repeats, 256–512 generated tokens and context >=131072")
    bins = smoke.validate_inputs(p, args)
    if args.prompt_file and not args.prompt_file.is_file():
        p.error("Missing --prompt-file")
    if args.r19_builds and not args.bundle:
        p.error("--r19-builds requires --bundle r19-compatible")
    if args.r19_builds:
        args.r19_builds = args.r19_builds.expanduser().resolve()
    treatments = args.profile or ([] if args.bundle else ["planner_only"])
    labels = list(dict.fromkeys(["baseline", "candidate_all_off", args.control] + treatments))
    if args.control.startswith("p16") or any(t.startswith("p16") for t in treatments):
        p.error("This benchmark fixes --spec-type none. Use p17_smoke --mtp for compatible drafters.")
    for t in treatments:
        diff = {k for k in profiles[t].keys() | profiles[args.control].keys()
                if profiles[t].get(k) != profiles[args.control].get(k)}
        if len(diff) != 1:
            p.error(f"{args.control} vs {t} differs in {len(diff)} flags. Compare adjacent profiles.")
    arms = {label: (bins["baseline" if label == "baseline" else "candidate"],
                    profiles["candidate_all_off" if label == "baseline" else label])
            for label in labels}
    if args.bundle:
        full19 = bundle_flags(profiles)
        full20 = bundle_flags(profiles, include_r20=True)
        if args.r19_builds:
            historical_bin = args.r19_builds / "candidate/bin/llama-server"
            arms["r19_all_off"] = (historical_bin, historical_flags(profiles["candidate_all_off"]))
            arms["r19_full"] = (historical_bin, historical_flags(full19))
        arms["r20_r19_full"] = (bins["candidate"], full19)
        arms["r20_full"] = (bins["candidate"], full20)
    labels = list(arms)
    smoke.new_output(p, args.out)
    path = args.out / "BENCHMARKS.json"
    order = crossover(labels, args.repeats, args.order_seed)
    report = {"schema_version": 1, "status": "RUNNING", "order": order,
              "selected_profiles": {k: env for k, (_, env) in arms.items()},
              "flag_deltas_from_all_off": {k: {key: v for key, v in env.items()
                                                if v != profiles["candidate_all_off"].get(key)}
                                           for k, (_, env) in arms.items()},
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "host": platform.uname()._asdict(), "runs": []}
    atomic_write(path, report)
    try:
        report["provenance"] = provenance(args)
        if args.r19_builds:
            report["r19_provenance"] = provenance(args, args.r19_builds, HISTORICAL_R19_MANIFEST)
        report["shards_dev_ino_bytes_mtime_ns"] = model_inventory(args.model)
        # Separate excluded conditioning pass for every variant before timed blocks.
        report["conditioning"] = []
        reference = None
        for block, variants in enumerate([order[0], *order], start=-1):
            for position, variant in enumerate(variants):
                stage = "condition" if block < 0 else f"b{block + 1:02}"
                label = f"{stage}-{position + 1:02}-{variant}"
                if model_inventory(args.model) != report["shards_dev_ino_bytes_mtime_ns"]:
                    raise ValueError("Model shard identity changed during the benchmark")
                report["active"] = label
                atomic_write(path, report)
                binary, env = arms[variant]
                with smoke.server(binary, label, env, args) as (port, launch):
                    warm = warmup(port, args)
                    if block < 0:
                        report["conditioning"].append({"variant": variant, "warmup": warm})
                        continue
                    ids = prompt_ids(port, args)
                    row = {"block": block, "position": position, "label": label, "variant": variant,
                           "startup_to_health_s": launch["startup_to_health_s"],
                           "offload_layers": launch["offload_layers"],
                           "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                           "prompt_tokens": len(ids), "warmup": warm,
                           **exact_completion(port, ids, args.predict, args.timeout)}
                    report["runs"].append(row)
                    atomic_write(path, report)
                    compare_ids(reference or row, row)
                    reference = reference or row
                    row["rss_peak_kib_through_decode"] = high_water_kib(launch["pid"])
                    row["gpu_after_decode"] = smoke.gpu_snapshot()
                    row["ttft_probe"] = completion_stream(port, ids, 8, args.timeout)
                    row["ttft_s"] = row["ttft_probe"]["ttft_s"]
                print(f"{label}   PP {row['timings']['prompt_per_second']:.3f}   "
                      f"TG {row['timings']['predicted_per_second']:.3f}   "
                      f"TTFT {row['ttft_s']:.3f} s", flush=True)
                atomic_write(path, report)
        report["summary"] = summarize_runs(report["runs"])
        pairs = [("baseline", "candidate_all_off")]
        for t in treatments:
            pairs.extend([(args.control, t), ("candidate_all_off", t), ("baseline", t)])
        if args.bundle:
            for treatment in ("r20_r19_full", "r20_full"):
                pairs.extend([("baseline", treatment), ("candidate_all_off", treatment)])
            pairs.append(("r20_r19_full", "r20_full"))
            if args.r19_builds:
                pairs.extend([("r19_all_off", "r19_full"), ("baseline", "r19_full"),
                              ("r19_all_off", "candidate_all_off"),
                              ("r19_full", "r20_r19_full"), ("r19_full", "r20_full")])
        report["paired_gains"] = paired_gains(report["runs"], list(dict.fromkeys(pairs)), args.repeats)
        report.update(status="COMPLETE_EQUAL_GREEDY_IDS")
        del report["active"]
        with (args.out / "SUMMARY.tsv").open("x") as table:
            table.write("variant\tmetric\tn\tmedian\tp95\tsample_variance\n")
            for variant, group in report["summary"].items():
                for metric, stats in group.items():
                    table.write(f"{variant}\t{metric}\t{stats['n']}\t{stats['median']:.8g}\t"
                                f"{stats['p95']:.8g}\t{stats['variance']:.8g}\n")
        for label, group in report["summary"].items():
            print(f"{label}: median PP {group['prompt_per_second']['median']:.3f}, "
                  f"median TG {group['predicted_per_second']['median']:.3f} tok/s")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, KeyboardInterrupt) as exc:
        report.update(status="FAIL", error=str(exc) or type(exc).__name__)
        print(report["error"], file=sys.stderr)
    finally:
        atomic_write(path, report)
    print(report["status"], path, flush=True)
    return int(report["status"] != "COMPLETE_EQUAL_GREEDY_IDS")


if __name__ == "__main__":
    raise SystemExit(main())
