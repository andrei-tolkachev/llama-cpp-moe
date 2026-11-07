#!/usr/bin/env python3
"""Compare greedy token IDs on two pinned servers, sequentially, over loopback."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

KIT = Path(__file__).resolve().parents[1]
PROMPT = "Москва — столица России. Берлин — столица"
FILL_TEXT = " The brown fox crosses a field and follows a quiet river."
API = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(port, route, body=None, timeout=10):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{route}", payload,
                                 headers={"Content-Type": "application/json"})
    with API.open(req, timeout=timeout) as response:
        return json.load(response)


def new_output(parser, path):
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {path}. Previous logs were preserved. "
                     "Choose a fresh --out directory, e.g. out-$(date +%Y%m%d-%H%M%S)-$$.")


def gpu_snapshot():
    """Driver-reported availability for each physical device at sampling time."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    try:
        result = {}
        for name, switch, fields in (
                ("gpu", "gpu", "index,uuid,pci.bus_id,name,memory.total,memory.free,memory.used,driver_version"),
                ("process", "compute-apps", "gpu_uuid,pid,used_memory")):
            cmd = [exe, f"--query-{switch}={fields}", "--format=csv,noheader,nounits"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            result[name] = {"fields": fields.split(","), "rows": proc.stdout.strip().splitlines(),
                            "error": proc.stderr.strip(), "exit_code": proc.returncode}
        return {"available": result["gpu"]["exit_code"] == 0, "memory_unit": "MiB", **result}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}


def log_diagnostic(logfile):
    # Retain the whole log on disk, but bound console and report error text.
    with logfile.open("rb") as src:
        size = src.seek(0, 2)
        src.seek(max(0, size - 65536))
        tail = src.read().decode(errors="replace").splitlines()
    clues = [line for line in tail if re.search(r"out of memory|cudaMalloc|failed to allocate|free.*MiB", line, re.I)]
    kind = "CUDA OOM" if any(re.search("out of memory", x, re.I) for x in clues) else "server failure"
    return kind + " | " + " | ".join((clues or tail)[-5:])[-1800:]


@contextmanager
def server(binary, label, env_flags, args, dspark_gguf=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LLAMA_EXPERIMENTAL_", "LLAMA_ARG_"))}
    env.update(env_flags, LD_LIBRARY_PATH=str(binary.parent) +
               (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""))
    alias = "p17-" + secrets.token_hex(12)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = [str(binary), "-m", str(args.model), "--host", "127.0.0.1", "--port", str(port),
           "--alias", alias, "-lv", "4", "-np", "1", "-c", str(args.ctx), "--fit", "off",
           "-ngl", str(args.ngl), "-b", "128", "-ub", "64", "-t", str(args.threads),
           "--cache-type-k", "f16", "--cache-type-v", "f16", "--no-warmup", "--no-ui",
           "--no-context-shift", "--offline"]
    if dspark_gguf is not None:
        if args.mtp:
            raise ValueError("Choose one trained draft mechanism per comparison")
        cmd.extend(["--spec-type", "draft-dspark", "--spec-draft-model", str(dspark_gguf),
                    "--spec-draft-ngl", "all", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0",
                    "--spec-draft-threads", str(args.threads)])
    elif args.mtp:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.draft_max)])
        if args.draft_model:
            cmd.extend(["--spec-draft-model", str(args.draft_model)])
    else:
        cmd.extend(["--spec-type", "none"])
    if args.moe_cache_mib:
        cmd.extend(["--cpu-moe", "--moe-cache-mib", str(args.moe_cache_mib)])
    logfile = args.out / (label + ".log")
    print(f"{label} — {logfile}", flush=True)
    record = {"command": cmd, "experiment_environment": env_flags, "gpu_before_launch": gpu_snapshot(),
              "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")}
    diagnostic_path = args.out / (label + ".launch.json")
    def save_record():
        diagnostic_path.write_text(json.dumps(record, indent=2) + "\n")
    save_record()
    with logfile.open("xb") as log:
        started = time.perf_counter()
        process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            record["pid"] = process.pid
            save_record()
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited {process.returncode}: {log_diagnostic(logfile)}. See {logfile} and {diagnostic_path}")
                try:
                    if request(port, "health")["status"] == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f"{label} load timed out. See {logfile}")
            record["startup_to_health_s"] = time.perf_counter() - started
            props = request(port, "props")
            if (props.get("model_alias") != alias or props.get("total_slots") != 1 or
                    Path(props["model_path"]).resolve() != args.model or
                    props["default_generation_settings"]["n_ctx"] != args.ctx):
                raise ValueError(f"Unexpected server model, alias, slot count or effective context; "
                                 f"reported model {props.get('model_path')}, slots {props.get('total_slots')}, "
                                 f"n_ctx {props.get('default_generation_settings', {}).get('n_ctx')}, expected {args.ctx}")
            record["effective_slot_ctx"] = props["default_generation_settings"]["n_ctx"]
            record["model_path"] = props["model_path"]
            record["model_ftype"] = props.get("model_ftype")
            with logfile.open("rb") as handle:
                offload = next((m for line in handle if (m := re.search(
                    rb'offloaded (\d+)/(\d+) layers to GPU', line))), None)
            if args.ngl and (offload is None or int(offload[1]) != min(args.ngl, int(offload[2]))):
                raise ValueError(f"GPU layer offload was not confirmed in {logfile}")
            record["offload_layers"] = list(map(int, offload.groups())) if offload else None
            record["gpu_after_load"] = gpu_snapshot()
            save_record()
            yield port, record
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            record["failure"] = str(exc)
            record["gpu_on_error"] = gpu_snapshot()
            clue = log_diagnostic(logfile)
            if "CUDA OOM" in clue and "CUDA OOM" not in str(exc):
                raise RuntimeError(f"{label}: {clue}. See {logfile} and {diagnostic_path}") from exc
            raise
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            record["exit_code"] = process.returncode
            record["gpu_after_exit"] = gpu_snapshot()
            save_record()
        if process.returncode:
            raise RuntimeError(f"{label} failed on shutdown (exit {process.returncode}). See {logfile}")


def smoke(binary, label, env_flags, args):
    with server(binary, label, env_flags, args) as (port, launch):
        prompt = request(port, "tokenize", {"content": PROMPT, "add_special": True})["tokens"]
        if args.fill_tokens:
            filler = request(port, "tokenize", {"content": FILL_TEXT, "add_special": False})["tokens"]
            if not filler or args.fill_tokens < len(prompt):
                raise ValueError("Requested prompt length too small, or tokenizer returned no filler")
            n = args.fill_tokens - len(prompt)
            prompt += (filler * ((n + len(filler) - 1) // len(filler)))[:n]
        res = request(port, "completion", {"prompt": prompt, "temperature": 0, "seed": 42,
                      "n_predict": args.predict, "return_tokens": True, "ignore_eos": True,
                      "cache_prompt": False, "stream": False}, args.timeout)
        ids = res["tokens"]
        if (not isinstance(ids, list) or not all(type(t) is int and t >= 0 for t in ids)
                or len(ids) != args.predict or res["tokens_predicted"] != len(ids)
                or res["tokens_evaluated"] != len(prompt) or res["truncated"]):
            raise ValueError(f"Invalid or truncated token result from {label}")
        if args.mtp and res["timings"].get("draft_n", 0) <= 0:
            raise ValueError(f"No MTP draft tokens were verified for {label}")
        result = {"label": label, "command": launch["command"], "experiment_environment": env_flags,
                "context": args.ctx, "prompt_tokens": len(prompt),
                "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                "tokens": ids, "content": res["content"], "timings": res["timings"]}
    return result


def validate_inputs(parser, args):
    args.model = args.model.expanduser().resolve()
    args.builds = args.builds.expanduser().resolve()
    if not args.model.is_file():
        parser.error("Model GGUF not found: " + str(args.model))
    shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", args.model.name)
    if shard:
        count = int(shard[2])
        prefix = args.model.name[:shard.start()]
        if shard[1] != "00001" or count < 1:
            parser.error("Specify the first shard")
        for i in range(1, count + 1):
            path = args.model.with_name(f"{prefix}-{i:05}-of-{count:05}.gguf")
            if not path.is_file():
                parser.error("Missing shard: " + str(path))
    if (min(args.ctx, args.threads, args.predict, args.timeout) < 1 or
            min(args.ngl, args.fill_tokens, args.moe_cache_mib) < 0 or
            args.fill_tokens + args.predict >= args.ctx):
        parser.error("Invalid sizes or prompt/output exceed context capacity")
    bins = {r: args.builds / r / "bin/llama-server" for r in ("baseline", "candidate")}
    if not all(p.is_file() and os.access(p, os.X_OK) for p in bins.values()):
        parser.error("Missing executable in baseline/bin or candidate/bin under --builds")
    return bins


def main():
    profiles = {p["name"]: p["environment"] for p in
                json.loads((KIT / "profiles/experiments.json").read_text())["profiles"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builds", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="GGUF or first shard")
    parser.add_argument("--out", type=Path, required=True, help="New report directory")
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--ngl", type=int, default=18)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--moe-cache-mib", type=int, default=12288)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--fill-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per model load or completion")
    parser.add_argument("--mtp", action="store_true", help="Use single MTP on baseline and candidate")
    parser.add_argument("--draft-model", type=Path, help="Matching MTP GGUF for a separate drafter")
    parser.add_argument("--draft-max", type=int, default=3)
    allowed = sorted(set(profiles) - {"candidate_all_off"})
    parser.add_argument("--profile", choices=allowed, help="Also test this candidate profile after all-off")
    args = parser.parse_args()
    if not args.mtp and (args.draft_model or (args.profile or "").startswith("p16")):
        parser.error("This draft model/profile requires --mtp")
    if args.draft_max < 1:
        parser.error("Baseline MTP comparison requires --draft-max >= 1")
    if args.draft_model:
        args.draft_model = args.draft_model.expanduser().resolve()
        if not args.draft_model.is_file():
            parser.error("Draft GGUF not found: " + str(args.draft_model))
    bins = validate_inputs(parser, args)
    new_output(parser, args.out)
    report = {"status": "RUNNING", "model": str(args.model), "runs": []}
    record = args.out / "smoke.json"
    schedule = [("baseline", "candidate_all_off"), ("candidate", "candidate_all_off")]
    if args.profile:
        schedule.append(("candidate", args.profile))
    try:
        for role, profile in schedule:
            run = smoke(bins[role], role if role == "baseline" else profile, profiles[profile], args)
            report["runs"].append(run)
            base = report["runs"][0]
            if run["prompt_ids_sha256"] != base["prompt_ids_sha256"]:
                raise ValueError(f"Tokenized prompts differ for {profile}. Compare {record}")
            if run["tokens"] != base["tokens"]:
                pos = next(i for i, (a, b) in enumerate(zip(base["tokens"], run["tokens"])) if a != b)
                raise ValueError(f"{profile}: output ID at position {pos} differs "
                                 f"({base['tokens'][pos]} vs {run['tokens'][pos]}). See {record}")
            timings = run["timings"]
            print(f"{run['label']}  prompt {run['prompt_tokens']} tokens / "
                  f"{timings['prompt_per_second']:.2f} t/s, generation {len(run['tokens'])} tokens / "
                  f"{timings['predicted_per_second']:.2f} t/s", flush=True)
            record.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report["status"] = "PASS_TOKEN_SMOKE"
    except (OSError, ValueError, RuntimeError, KeyError, KeyboardInterrupt) as exc:
        report.update(status="FAIL", error=str(exc) or type(exc).__name__)
        print(report["error"], file=sys.stderr)
    finally:
        record.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['status']}  {record}")
    return int(report["status"] != "PASS_TOKEN_SMOKE")


if __name__ == "__main__":
    sys.exit(main())
