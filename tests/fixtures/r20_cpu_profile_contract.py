#!/usr/bin/env python3
"""Genuine localhost CPU profiling on a manual build. Only the production receipt lookup is stubbed."""
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT / "tools"))
import r20_profile as profiler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", type=Path, required=True, help="Manually built CPU server")
    p.add_argument("--model", type=Path, required=True, help="stories260K.gguf")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    server = args.server.resolve(strict=True)
    provenance = {"scope": "PROTOCOL_FIXTURE_MANUAL_CPU_BUILD",
                  "receipt_mocked": True, "server": str(server),
                  "server_sha256": profiler.bench.file_hash(server),
                  "model_sha256": profiler.bench.file_hash(args.model)}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        # validate_inputs expects a pair; diagnostics launch only the candidate.
        for role in ("baseline", "candidate"):
            (root/role).mkdir()
            (root/role/"bin").symlink_to(server.parent, target_is_directory=True)
        cli = ["r20_profile.py", "--builds", str(root), "--model", str(args.model), "--out", str(args.out),
               "--ctx", "512", "--ngl", "0", "--threads", "1", "--moe-cache-mib", "0",
               "--profile", "r20-full", "--predict", "256", "--perf", "--devices"]
        print(f"CPU protocol fixture. Receipt lookup only mocked. Server {server}.", flush=True)
        with patch.object(sys, "argv", cli), patch.object(profiler.bench, "provenance", return_value=provenance):
            return profiler.main()


if __name__ == "__main__":
    sys.exit(main())
