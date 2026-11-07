#!/usr/bin/env python3
"""Disconnect a streamed request on an isolated candidate, verify early cancel and reuse."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

import p17_smoke as smoke
import r20_bench as bench


def disconnect(port, ids, predict, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 json.dumps(bench.payload(ids, predict, True)).encode(),
                                 headers={"Content-Type": "application/json"})
    with smoke.API.open(req, timeout=timeout) as resp:
        for line in resp:
            if line.startswith(b"data:"):
                row = json.loads(line[5:])
                if row.get("tokens") and not row.get("stop"):
                    return row["tokens_predicted"]  # closes this response on return
    raise ValueError("Stream ended without a token to interrupt")


def cancel_and_reuse(port, launch, args, logfile):
    prompt = bench.prompt_ids(port, args)
    before = bench.exact_completion(port, prompt, args.predict, args.timeout)
    stream_tokens = min(4096, args.ctx - len(prompt) - 1)
    if stream_tokens <= args.predict:
        raise ValueError("Not enough remaining context for cancellation probe")
    first = disconnect(port, prompt, stream_tokens, args.timeout)
    started = time.monotonic()
    while time.monotonic() - started < args.cancel_timeout:
        slots = smoke.request(port, "slots", timeout=min(10, args.cancel_timeout))
        if len(slots) != 1:
            raise ValueError("Unexpected slot count")
        slot = slots[0]
        if not slot["is_processing"]:
            break
        time.sleep(.05)
    else:
        raise TimeoutError("Slot failed to return to idle after disconnect")
    idle = time.monotonic() - started
    # The pinned server logs the cancel task and preserves the previous task ID on an idle slot.
    marker = re.compile(rb"cancel task, id_task = " + str(slot["id_task"]).encode() + rb"\b")
    with logfile.open("rb") as f:
        cancelled = any(marker.search(line) for line in f)
    n_remaining = slot["n_prompt_tokens"]
    if not cancelled or n_remaining >= len(prompt) + stream_tokens - 1:
        raise ValueError(f"Early cancel not confirmed: cancel_log={cancelled}, cached_tokens={n_remaining}")
    after = [bench.exact_completion(port, prompt, args.predict, args.timeout) for _ in range(2)]
    result = {"context": args.ctx, "pid": launch["pid"], "prompt_tokens": len(prompt),
              "requested_stream_tokens": stream_tokens, "client_tokens_at_close": first,
              "slot_idle_after_close_s": idle, "cancelled_task_id": slot["id_task"],
              "idle_slot_prompt_tokens": n_remaining, "before": before, "after": after}
    result["same_ids"] = all(r["tokens"] == before["tokens"] for r in after)
    return result


def main():
    profiles = {p["name"]: p["environment"] for p in
                json.loads((smoke.KIT / "profiles/experiments.json").read_text())["profiles"]}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--builds", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--ngl", type=int, default=18)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--moe-cache-mib", type=int, default=12288)
    p.add_argument("--predict", type=int, default=64)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--cancel-timeout", type=int, default=120)
    p.add_argument("--profile", choices=[k for k in profiles if not k.startswith("p16")], default="candidate_all_off")
    args = p.parse_args()
    args.fill_tokens, args.prompt_file, args.mtp = 0, None, False
    bins = smoke.validate_inputs(p, args)
    smoke.new_output(p, args.out)
    report = {"status": "RUNNING", "profile": args.profile}
    path = args.out / "cancel.json"
    try:
        report["provenance"] = bench.provenance(args)
        with smoke.server(bins["candidate"], "cancel", profiles[args.profile], args) as (port, launch):
            report["result"] = cancel_and_reuse(port, launch, args, args.out / "cancel.log")
            if not report["result"]["same_ids"]:
                raise ValueError("Post-cancel request produced different IDs")
        report["status"] = "PASS_CANCEL_REUSE_SHUTDOWN"
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        report.update(status="FAIL", error=str(exc) or type(exc).__name__)
    finally:
        bench.atomic_write(path, report)
    print(report["status"], report.get("error", ""), path)
    return int(report["status"] != "PASS_CANCEL_REUSE_SHUTDOWN")


if __name__ == "__main__":
    sys.exit(main())
