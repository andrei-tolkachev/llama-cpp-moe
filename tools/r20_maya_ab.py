#!/usr/bin/env python3
"""Cross-engine, same-shard benchmark of an isolated Project Maya and R20; Linux only."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

import p17_smoke as smoke
import r20_bench as bench
from r20_gguf_inventory import Gguf

MAYA_PIN = "3589d6f12a3f77e813b184b51a586f8fc3de2185"
MAYA_ENV = {"DS4_YOCO": "0", "STRATA_DS4_DISK_SKIP_W": "0", "STRATA_DS4_RAM_SKIP_W": "0",
            "DS4_BENCH_REPEAT": "2", "STRATA_DS4_LOAD_DROP_CACHE": "0", "DS4_COPY_DROP_CACHE": "0"}


def checked(cmd, path, timeout, env=None, data=None):
    with path.open("xb") as f:
        proc = subprocess.run(cmd, input=data, stdin=subprocess.DEVNULL if data is None else None,
                              stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if proc.returncode:
        raise ValueError(f"Command exited {proc.returncode}; see {path}")
    return path.read_text(errors="replace")


def validate_pack(pack, alias, shards, drafter):
    geo = (pack / "ds4.txt").read_text()
    if "format gguf" not in geo.splitlines() or f"model_dir {alias}" not in geo.splitlines():
        raise ValueError("Packer changed model directory or format")
    allowed = set(shards)
    external = drafter.with_suffix(".bin").resolve() if drafter else None
    files = {}
    with (pack / "tensors.txt").open() as f:
        for line in f:
            cols = line.split()
            if cols[0] == "file":
                fi = int(cols[1])
                path = (alias / " ".join(cols[2:])).resolve()
                if fi in files or (path not in allowed and path != external):
                    raise ValueError(f"Duplicate or unexpected source in private pack: {path}")
                files[fi] = path
            elif cols[0] == "t":
                path = files[int(cols[3])]
                if path == external and not cols[1].startswith("mtp."):
                    raise ValueError("Trunk tensor would be sourced from the drafter")
    expected = allowed | ({external} if external else set())
    if set(files.values()) != expected:
        raise ValueError("Private pack did not reference exactly the requested GGUF shards and drafter")
    return {"file_table": {i: str(v) for i, v in files.items()}, "geometry_sha256": bench.file_hash(pack/"ds4.txt"),
            "tensor_index_sha256": bench.file_hash(pack/"tensors.txt")}


def prepare_pack(args, inventory):
    # Auto-discovery in the donor packer can silently substitute neighboring Engram files.
    alias = args.out / "model-view"
    alias.mkdir()
    shards = [Path(name) for name in inventory]
    for path in shards:
        (alias/path.name).symlink_to(path)
    pack = args.out / "maya-pack"
    tool = args.maya_repo / "tools/ds4/pack_gguf.py"
    cmd = [sys.executable, "-I", "-B", str(tool), "--model", str(alias), "--out", str(pack),
           "--shards", *[p.name for p in shards]]
    if args.drafter:
        cmd.extend(["--drafter", str(args.drafter)])
    output = checked(cmd, args.out/"pack.log", args.timeout)
    if "Engram tables from" in output or re.search(r"Engram .*over the quant", output):
        raise ValueError("External Engram table/metadata rejected")
    return pack, {"command": cmd, **validate_pack(pack, alias, shards, args.drafter)}


def maya_flags(args, pack, spec):
    return [str(args.maya_bin), "--pack", str(pack), "--gpus", "0", "--tp", "1",
            "--max-context", str(args.ctx), "--cpu-threads", str(args.threads), "--parallel", "1",
            "--reserve-mib", str(args.reserve_mib), "--mmap-experts", "auto", "--threads", "on",
            "--ssd-cache", args.maya_ssd_cache, "--expert-profile", "none", "--profile-readonly",
            "--suffix-draft", "0", "--dspark", "on" if spec else "off", "--vision", "off",
            "--temperature", "0", "--seed", "42", "--prefill-chunk", "64", "--window", "64", "--verbose"]


def parse_maya(stdout, stderr, n_input, predict, spec):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 2 or any(not re.fullmatch(r"(?:\d+,)+", line) for line in lines):
        raise ValueError("Expected exactly two complete numeric ID sequences on Maya stdout")
    logs = stderr.split("pass 2 of 2\n")
    if len(logs) != 2:
        raise ValueError("Maya did not mark a second pass after reset")
    passes = []
    for text, log in zip(lines, logs):
        ids = [int(x) for x in text.rstrip(",").split(",")]
        pp = re.search(r"^prompt: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s,", log, re.M)
        tg = re.search(r"^decode: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\); CPU lane: (\d+) of (\d+) experts, streamed to the GPU: (\d+)", log, re.M)
        if (pp is None or tg is None or int(pp[1]) != n_input or int(tg[1]) != len(ids)
                or not 1 <= len(ids) <= predict or (len(ids) < predict and ids[-1] != 1)
                or float(pp[3]) <= 0 or (len(ids) > 1 and float(tg[3]) <= 0)):
            raise ValueError("Maya input/output count or timing protocol mismatch")
        drafts = re.search(r"^dspark: (\d+) rounds, (\d+) drafts offered, (\d+) accepted", log, re.M)
        offered, accepted = (int(drafts[2]), int(drafts[3])) if drafts else (0, 0)
        if accepted > offered or (not spec and offered > 0):
            raise ValueError("Unexpected DSpark counters")
        passes.append({"tokens": ids, "native_eog": ids[-1] == 1, "timings": {
            "prompt_n": int(pp[1]), "predicted_n": int(tg[1]), "prompt_ms": float(pp[2]),
            "predicted_ms": float(tg[2]), "prompt_per_second": float(pp[3]), "predicted_per_second": float(tg[3])},
            "cpu_experts": int(tg[4]), "total_experts": int(tg[5]), "streamed_experts": int(tg[6]),
            "drafts_offered": offered, "drafts_accepted": accepted})
    return passes


def maya_run(args, pack, label, ids, env, spec):
    cmd = maya_flags(args, pack, spec) + ["--ids", "@" + str(args.out/"prompt.ids"), "--n", str(args.predict)]
    out, err = args.out/(label+".stdout"), args.out/(label+".stderr")
    before = smoke.gpu_snapshot()
    start = time.monotonic()
    with out.open("xb") as stdout, err.open("xb") as stderr:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env, timeout=args.timeout)
    elapsed = time.monotonic() - start
    if proc.returncode:
        raise ValueError(f"Maya exited {proc.returncode}; see {err}")
    log = err.read_text(errors="replace")
    passes = parse_maya(out.read_text(), log, len(ids), args.predict, spec)
    if "re-quantized" in log or "decoding with prompt lookup only" in log:
        raise ValueError(f"Maya reports altered weights or missing requested DSpark. See {err}")
    return {"command": cmd, "gpu_before": before, "process_total_s": elapsed, "warmup": passes[0], **passes[1],
            "placement": [line for line in log.splitlines() if line.startswith("ds4: CUDA") or
                          ("routed experts on the CPU lane" in line and line.startswith("ds4:"))]}


def trained_draft_identity(path):
    # Same DFLASH + Markov-head discriminator as the pinned common/speculative.cpp.
    try:
        g = Gguf(path)
    except (SystemExit, struct.error) as exc:
        raise ValueError(f"Invalid trained DSpark GGUF: {path}") from exc
    if (g.kv.get("general.architecture") != "dflash" or g.kv.get("split.count", 1) != 1
            or not any(name == "markov_w1.weight" for name, *_ in g.tensors)):
        raise ValueError("Expected a single trained DSpark GGUF (dflash architecture with markov_w1.weight)")
    return {"path":str(path), "sha256":bench.file_hash(path), "format":"DFLASH GGUF / DSpark Markov head",
            "model_name":g.kv.get("general.name"), "block_size":g.kv.get("dflash.block_size")}


def draft_counts(timings, requested):
    offered = timings.get("draft_n", 0)
    accepted = timings.get("draft_n_accepted", 0)
    if (type(offered) is not int or type(accepted) is not int or
            not 0 <= accepted <= offered or (not requested and offered > 0)):
        raise ValueError("Invalid llama draft counters or active draft on a plain arm")
    return {"drafts_offered":offered, "drafts_accepted":accepted}


def llama_run(args, binary, label, ids, env, draft=None):
    def native(port):
        req = bench.payload(ids, args.predict)
        req["ignore_eos"] = False
        r = smoke.request(port, "completion", req, args.timeout)
        tokens = r["tokens"]
        n = len(tokens)
        if not 1 <= n <= args.predict or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid native token sequence")
        if n < args.predict and r["stop_type"] != "eos":
            raise ValueError("Unexpected short native generation")
        # At native EOG on token 1, TG has zero decode steps.
        if n > 1:
            bench.validate_completion(r, len(ids), n)
        elif (r["truncated"] or r["tokens_evaluated"] != len(ids) or r["timings"]["prompt_n"] != len(ids)
              or r["timings"]["cache_n"] != 0 or r["tokens_predicted"] != 1 or r["timings"]["predicted_n"] != 1):
            raise ValueError("Invalid one-token completion")
        return {"tokens": tokens, "native_eog": r["stop_type"] == "eos", "timings": r["timings"],
                **draft_counts(r["timings"], draft is not None)}
    with smoke.server(binary, label, env, args, dspark_gguf=draft) as (port, launch):
        warm, result = native(port), native(port)
        return {"command": launch["command"], "warmup": warm, **result, "startup_s": launch["startup_to_health_s"],
                "offload_layers": launch["offload_layers"], "rss_peak_kib": bench.high_water_kib(launch["pid"])}


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def main():
    profiles = {x["name"]: x["environment"] for x in json.loads((smoke.KIT/"profiles/experiments.json").read_text())["profiles"]}
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "builds", "out", "maya-repo", "maya-bin"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--r19-builds", type=Path)
    p.add_argument("--maya-ssd-cache", choices=("off","auto"), default="off", help="Optional second run with Maya RAM expert store; adaptive VRAM cache stays enabled")
    p.add_argument("--llama-dspark-gguf", type=Path, help="Single compatible trained DFLASH Markov-head GGUF for llama-side DSpark arms")
    p.add_argument("--drafter", type=Path, help="Compatible trained dspark.json with matching .bin")
    p.add_argument("--drafter-bin-sha256", help="Expected hash from trusted pack provenance")
    p.add_argument("--profile", choices=["candidate_all_off", "r20_full", "planner_only", "cpu_iq3_signmask"],
                   default="r20_full")
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--fill-tokens", type=int, default=0)
    p.add_argument("--order-seed", type=int, default=20261103)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--predict", type=int, default=256)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--reserve-mib", type=int, default=3072)
    p.add_argument("--timeout", type=int, default=7200)
    args = p.parse_args()
    args.ngl, args.moe_cache_mib, args.mtp = 18, 12288, False
    if args.repeats < 5 or args.ctx < 131072 or not 256 <= args.predict <= 512 or args.reserve_mib < 0:
        p.error("Need >=5 repeats, >=131072 context and 256–512 predicted tokens")
    if bool(args.drafter) != bool(args.drafter_bin_sha256):
        p.error("--drafter and --drafter-bin-sha256 must be set together")
    bins = smoke.validate_inputs(p, args)
    args.out, args.maya_bin, args.maya_repo = (x.expanduser().resolve() for x in (args.out, args.maya_bin, args.maya_repo))
    smoke.new_output(p, args.out)
    path = args.out/"MAYA_AB.json"
    report = {"status": "RUNNING", "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "maya_env_explicit": MAYA_ENV, "runs": [], "gguf": str(args.model),
              "comparison": "same input/shards, engine-native KV, kernels and expert placement; native EOG"}
    # Maya explicitly uses its CUDA0; expose the same single device to both child families.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES","0")
    if not visible or "," in visible or visible == "-1":
        p.error("Cross-engine protocol needs one CUDA device. Set CUDA_VISIBLE_DEVICES to its index or GPU UUID.")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    report["cuda_visible_devices"] = visible
    env = {k:v for k,v in os.environ.items() if not k.startswith(("STRATA_", "DS4_", "MAYA_"))}
    env.update(MAYA_ENV)
    try:
        report["r20_build"] = bench.provenance(args)
        git = ["git", "-C", str(args.maya_repo)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if head != MAYA_PIN or subprocess.run([*git, "diff", "--quiet", "HEAD", "--"]).returncode:
            raise ValueError("Maya checkout must be a clean tracked tree at pinned 3589d6f12a3f")
        version = checked([str(args.maya_bin), "--version"], args.out/"maya-version.log", 30, env).strip()
        if not re.fullmatch(r"maya-ds4 1\.1\.46 \(build (?:3589d6f12a3f|"+MAYA_PIN+r")\)", version):
            raise ValueError("Wrong Maya reported build: "+version)
        report.update(maya_pin=head, maya_version=version, maya_binary_sha256=bench.file_hash(args.maya_bin))
        inventory = bench.model_inventory(args.model)
        report["shards_dev_ino_bytes_mtime_ns"] = inventory
        if args.llama_dspark_gguf:
            args.llama_dspark_gguf = args.llama_dspark_gguf.expanduser().resolve(strict=True)
            if str(args.llama_dspark_gguf) in inventory:
                raise ValueError("Trunk and trained GGUF drafter must be separate files")
            report["llama_drafter"] = trained_draft_identity(args.llama_dspark_gguf)
        if args.drafter:
            args.drafter = args.drafter.expanduser().resolve()
            digest = bench.file_hash(args.drafter.with_suffix(".bin"))
            if digest != args.drafter_bin_sha256:
                raise ValueError("Drafter binary SHA256 mismatch")
            report["drafter"] = {"json": str(args.drafter), "json_sha256": bench.file_hash(args.drafter), "bin_sha256": digest}
        pack, report["private_pack"] = prepare_pack(args, inventory)
        checked([str(args.maya_bin), "--pack", str(pack), "--pack-info"], args.out/"pack-info.log", args.timeout, env)
        ready = checked(maya_flags(args, pack, False)+["--serve"], args.out/"maya-context.log",
                        args.timeout, env, b"QUIT\n")
        if not re.search(rf"^READY {args.ctx} stop$", ready, re.M):
            raise ValueError("Effective Maya context not confirmed in maya-context.log")
        report["maya_info"] = [line for line in ready.splitlines() if line.startswith("INFO ")]
        with smoke.server(bins["baseline"], "tokenizer", profiles["candidate_all_off"], args) as (port, _):
            ids = bench.prompt_ids(port, args)
        (args.out/"prompt.ids").write_text(",".join(map(str, ids)))
        report["prompt_sha256"] = bench.file_hash(args.out/"prompt.ids")
        arms = {"baseline": (bins["baseline"], profiles["candidate_all_off"]),
                "r20": (bins["candidate"], bench.bundle_flags(profiles, True) if args.profile=="r20_full" else profiles[args.profile])}
        if args.r19_builds:
            report["r19_build"] = bench.provenance(args, args.r19_builds, bench.HISTORICAL_R19_MANIFEST)
            arms["r19_full"] = (args.r19_builds/"candidate/bin/llama-server", bench.historical_flags(bench.bundle_flags(profiles)))
        if args.llama_dspark_gguf:
            arms["r20_dspark"] = arms["r20"]
            if args.r19_builds:
                arms["r19_full_dspark"] = arms["r19_full"]
        labels = list(arms) + ["maya_plain"] + (["maya_dspark"] if args.drafter else [])
        report["order"] = bench.crossover(labels, args.repeats, args.order_seed)
        for i, block in enumerate(report["order"]):
            for j, variant in enumerate(block):
                if bench.model_inventory(args.model) != inventory:
                    raise ValueError("Model changed during benchmark")
                label = f"b{i+1:02}-{j+1:02}-{variant}"
                print(label, flush=True)
                report["active"] = label
                bench.atomic_write(path, report)
                if variant in arms:
                    binary, flags = arms[variant]
                    row = llama_run(args, binary, label, ids, flags,
                                    args.llama_dspark_gguf if variant.endswith("_dspark") else None)
                else:
                    row = maya_run(args, pack, label, ids, env, variant == "maya_dspark")
                row.update(block=i, variant=variant, label=label,
                           warmup_first_difference=first_difference(row["warmup"]["tokens"], row["tokens"]))
                report["runs"].append(row)
                bench.atomic_write(path, report)
        controls = {r["block"]: r for r in report["runs"] if r["variant"]=="baseline"}
        by_arm = defaultdict(list)
        for r in report["runs"]:
            r["first_difference_vs_baseline"] = first_difference(r["tokens"], controls[r["block"]]["tokens"])
            by_arm[r["variant"]].append(r)
        incomplete = False
        for variant, rows in by_arm.items():
            for r in rows:
                r["first_difference_vs_first_in_arm"] = first_difference(rows[0]["tokens"], r["tokens"])
            usable = [r for r in rows if len(r["tokens"]) == args.predict and
                      (not variant.endswith("_dspark") or r["drafts_offered"] > 0)]
            report.setdefault("summary", {})[variant] = {"valid_length_and_draft_n": len(usable),
                "native_eog_count": sum(r["native_eog"] for r in rows),
                "first_difference_by_block": [r["first_difference_vs_baseline"] for r in rows],
                "descriptive_full_length_only": {k: bench.summarize([r["timings"][k] for r in usable])
                   for k in ("prompt_per_second","predicted_per_second")} if usable else None}
            incomplete |= len(usable) != len(rows)
        same = all(r["first_difference_vs_baseline"] is None and r["warmup_first_difference"] is None and
                   r["first_difference_vs_first_in_arm"] is None for r in report["runs"])
        report["all_ids_equal"] = same
        pairs = [("baseline", "r20"), ("r20","maya_plain")]
        if args.llama_dspark_gguf:
            pairs.append(("r20","r20_dspark"))
            if args.r19_builds:
                pairs.extend([("r19_full","r19_full_dspark"), ("r19_full_dspark","r20_dspark")])
            if args.drafter:
                pairs.append(("r20_dspark","maya_dspark"))
        if args.drafter:
            pairs += [("maya_plain","maya_dspark"),("r20","maya_dspark")]
        if args.r19_builds:
            pairs.append(("r19_full","maya_plain"))
        valid_pairs = [pair for pair in pairs if all(report["summary"][v]["valid_length_and_draft_n"] == args.repeats
                                                       for v in pair)]
        report["paired_gains"] = bench.paired_gains(report["runs"], valid_pairs, args.repeats)
        report["pairs_skipped_for_short_output_or_zero_drafts"] = [pair for pair in pairs if pair not in valid_pairs]
        report["status"] = ("INCONCLUSIVE_LENGTH_OR_NO_DRAFT" if incomplete else
                            "COMPLETE_EQUAL_IDS" if same else "COMPLETE_DIFFERENT_IDS")
        del report["active"]
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        report.update(status="FAIL", error=str(exc) or type(exc).__name__)
    finally:
        bench.atomic_write(path, report)
    print(report["status"], report.get("error", ""), path)
    return int(report["status"] != "COMPLETE_EQUAL_IDS")


if __name__ == "__main__":
    sys.exit(main())
