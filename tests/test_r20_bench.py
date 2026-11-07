from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "tools"))
import r20_bench as bench


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.builds = self.work / "builds"
        arts = {}
        for role in ["baseline", "candidate"]:
            exe = self.builds / role / "bin/llama-server"
            exe.parent.mkdir(parents=True)
            shutil.copyfile(KIT / "tests/fixtures/r20_fake_server.py", exe)
            exe.chmod(0o700)
            arts[role] = {"artifacts": [{"path": "bin/llama-server", "sha256": bench.file_hash(exe)}]}
        pin = json.loads((KIT / "manifest.json").read_text())["upstream"]["base_commit"]
        receipt = {"status": "BUILD_ONLY_KNOWN_BASE", "exit_code": 0, "builds": arts,
                   "plan": {"cuda_requested": True, "cuda_arch_requested": "120",
                            "verification": {"base_commit": pin, "manifest_sha256": bench.file_hash(KIT / "manifest.json")}}}
        (self.builds / "build_record.json").write_text(json.dumps(receipt))
        self.model = self.work / "fake.gguf"
        self.model.touch()
        self.out = self.work / "out"
        self.argv = ["--builds", str(self.builds), "--model", str(self.model), "--out", str(self.out),
                     "--repeats", "5", "--timeout", "8"]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.sleep = time.sleep

    def run_bench(self, mode=""):
        with patch.dict(os.environ, R20_TEST_MODE=mode, R20_TEST_LOCK=str(self.work / "lock")), \
                patch("p17_smoke.time.sleep", lambda delay: self.sleep(.015)), \
                patch("p17_smoke.gpu_snapshot", return_value={"available": False}), \
                redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return bench.main(self.argv)

    def report(self):
        return json.loads((self.out / "BENCHMARKS.json").read_text())

    def test_server_crossover_and_separate_ttft(self):
        self.argv.extend(["--fill-tokens", "32768"])
        self.assertEqual(self.run_bench(), 0, self.stderr.getvalue())
        result = self.report()
        self.assertEqual(result["status"], "COMPLETE_EQUAL_GREEDY_IDS")
        self.assertEqual(len(result["runs"]), 15)
        self.assertEqual(len(result["conditioning"]), 3)
        self.assertEqual(result["order"][0], list(reversed(result["order"][1])))
        self.assertEqual(result["order"][2], list(reversed(result["order"][3])))
        self.assertEqual(len(result["paired_gains"]), 3)
        for r in result["runs"]:
            self.assertEqual(r["prompt_tokens"], 32768)
            self.assertEqual(len(r["tokens"]), 256)
            self.assertEqual(r["ttft_probe"]["emitted_ids_count"], 7)
            self.assertGreater(r["ttft_s"], 0)
        self.assertEqual(len(list(self.out.glob("*.log"))), 18)
        self.assertEqual(result["summary"]["baseline"]["predicted_per_second"]["variance"], 0)

    def test_token_mismatch_preserves_raw_runs(self):
        self.assertEqual(self.run_bench("mismatch"), 1)
        result = self.report()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("index 255", result["error"])
        self.assertGreaterEqual(len(result["runs"]), 2)

    def test_oom_and_existing_output(self):
        self.assertEqual(self.run_bench("oom"), 1)
        result = self.report()
        self.assertIn("CUDA OOM", result["error"])
        self.assertIn("3475 MiB", result["error"])
        self.assertTrue(list(self.out.glob("*.launch.json")))
        with self.assertRaises(SystemExit):
            self.run_bench()
        self.assertIn("Output already exists", self.stderr.getvalue())
        self.assertEqual(self.report(), result)

    def test_tampered_build_not_launched(self):
        with (self.builds / "candidate/bin/llama-server").open("a") as f:
            f.write("# after compilation")
        self.assertEqual(self.run_bench(), 1)
        self.assertIn("changed executable", self.report()["error"])
        self.assertFalse(list(self.out.glob("*.log")))

    def test_cached_or_truncated_prefill_does_not_count(self):
        for changes in [{"cache_n": 12, "prompt_n": 20}, {"cache_n": 0, "prompt_n": 31}]:
            with self.assertRaisesRegex(ValueError, "Cached, truncated"):
                bench.validate_completion({"truncated": False, "tokens_evaluated": 32, "tokens_predicted": 256,
                                           "timings": {**changes, "predicted_n": 256}}, 32, 256)

    def test_historical_bundle_uses_the_old_executable(self):
        historic = self.work / "r19-builds"
        shutil.copytree(self.builds, historic)
        path = historic / "build_record.json"
        record = json.loads(path.read_text())
        record["plan"]["verification"]["manifest_sha256"] = bench.HISTORICAL_R19_MANIFEST
        path.write_text(json.dumps(record))
        self.argv.extend(["--r19-builds", str(historic), "--bundle", "r19-compatible"])
        self.assertEqual(self.run_bench(), 0, self.stderr.getvalue())
        r = self.report()
        self.assertEqual(len(r["runs"]), 30)
        self.assertEqual(r["selected_profiles"]["r19_full"], bench.historical_flags(r["selected_profiles"]["r20_r19_full"]))
        self.assertEqual(len(r["selected_profiles"]["r19_full"]), 19)
        self.assertNotIn("LLAMA_EXPERIMENTAL_MOE_DEMAND_READAHEAD", r["selected_profiles"]["r19_full"])
        full = r["flag_deltas_from_all_off"]["r20_full"]
        self.assertEqual(full["LLAMA_EXPERIMENTAL_CPU_IQ3_SIGNMASK"], "1")
        self.assertEqual(full["LLAMA_EXPERIMENTAL_MOE_DEMAND_READAHEAD"], "1")
        self.assertEqual(full["LLAMA_EXPERIMENTAL_ENGRAM_ASYNC"], "1")
        self.assertNotIn("LLAMA_EXPERIMENTAL_CUDA_MMQ_PROFILE", full)
        self.assertNotIn("LLAMA_EXPERIMENTAL_MTP_REWIND_GUARD", full)
        old = next(v for v in r["runs"] if v["variant"] == "r19_full")
        launch = json.loads((self.out / (old["label"] + ".launch.json")).read_text())
        self.assertEqual(Path(launch["command"][0]), historic / "candidate/bin/llama-server")

    def test_historical_bundle_rejects_new_receipt(self):
        self.argv.extend(["--r19-builds", str(self.builds), "--bundle", "r19-compatible"])
        self.assertEqual(self.run_bench(), 1)
        self.assertFalse(list(self.out.glob("*.log")))
        self.assertEqual(self.report()["status"], "FAIL")

    def test_multiflag_profile_rejected_without_prerequisite(self):
        self.argv.extend(["--profile", "pipeline_only"])
        with self.assertRaises(SystemExit):
            self.run_bench()
        self.assertIn("differs in 2 flags", self.stderr.getvalue())
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
