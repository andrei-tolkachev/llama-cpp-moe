"""Regress the acceptance runner's failure gates without loading a model."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "p17_smoke", Path(__file__).resolve().parents[1] / "tools/p17_smoke.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class P17SmokeTests(unittest.TestCase):
    def test_profile_stops_at_first_token_mismatch_and_preserves_both_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role in ("baseline", "candidate"):
                exe = root / role / "bin/llama-server"
                exe.parent.mkdir(parents=True)
                exe.touch(mode=0o700)
            model = root / "model.gguf"
            model.touch()
            out = root / "output"
            args = ["p17", "--builds", tmp, "--model", str(model), "--out", str(out),
                    "--profile", "planner_only"]
            with patch.object(sys, "argv", args), patch.object(SMOKE, "smoke", side_effect=[
                    {"tokens": [1, 2], "prompt_ids_sha256": "same", "label": "baseline",
                     "prompt_tokens": 4, "timings": {"prompt_per_second": 10, "predicted_per_second": 8}},
                    {"tokens": [1, 3], "prompt_ids_sha256": "same"}]) as run:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(SMOKE.main(), 1)
            report = json.loads((out / "smoke.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual([r["tokens"] for r in report["runs"]], [[1, 2], [1, 3]])
            self.assertEqual(run.call_count, 2)

    def test_pipeline_profile_isolates_upload_from_staging_and_persistent_worker(self):
        profiles = {p["name"]: p["environment"] for p in
                    json.loads((SMOKE.KIT / "profiles/experiments.json").read_text())["profiles"]}
        for before, after, expected in (
                ("staging_only", "pipeline_only", "LLAMA_EXPERIMENTAL_MOE_PIPELINE"),
                ("pipeline_only", "persistent_worker_only", "LLAMA_EXPERIMENTAL_MOE_PERSISTENT")):
            diff = {k for k in profiles[before] if profiles[before][k] != profiles[after][k]}
            self.assertEqual(diff, {expected})

    def test_incomplete_shards_refused_before_server_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model-00001-of-00002.gguf"
            model.touch()
            out = root / "output"
            args = ["p17", "--builds", tmp, "--model", str(model), "--out", str(out)]
            err = io.StringIO()
            with patch.object(sys, "argv", args), redirect_stderr(err), self.assertRaises(SystemExit):
                SMOKE.main()
            self.assertIn("model-00002-of-00002.gguf", err.getvalue())
            self.assertFalse(out.exists())
