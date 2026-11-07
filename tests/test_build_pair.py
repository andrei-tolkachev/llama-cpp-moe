"""Safety/integrity tests with tiny local Git repositories and a fake CMake.

No model, GPU, network or actual compiler is used. Run with unittest discovery.
"""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "tools/build_pair.py"
SPEC = importlib.util.spec_from_file_location("build_pair_under_test", MODULE)
bp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bp)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class BuildPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llama pair tests ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kit = self.root / "kit"
        self.patches = self.kit / "patches"
        self.patches.mkdir(parents=True)
        self.origin = self.root / "origin"
        self.origin.mkdir()
        bp.git(self.origin, ["-c", "init.templateDir=", "init", "--quiet"])
        # A middle hunk allows a fresh upstream to change a distant prefix
        # without weakening git-apply's exact context checks.
        self.before = (b"// fixture\n" + b"// unchanged prefix\n" * 8 +
                       b"int value() {\n    return 1;\n}\n" + b"// unchanged suffix\n" * 8)
        self.after = self.before.replace(b"return 1;", b"return 2;")
        (self.origin / "src").mkdir()
        (self.origin / "src/target.cpp").write_bytes(self.before)
        (self.origin / "README").write_text("fixture\n")
        (self.origin / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.18)\nproject(fixture)\n' +
            "\n".join('option(' + key + ' "fixture" OFF)' for key in [*bp.PROJECT_OPTIONS, "GGML_CUDA"]))
        (self.origin / "ggml/src/ggml-cuda").mkdir(parents=True)
        (self.origin / "ggml/CMakeLists.txt").write_text("# fixture\n")
        (self.origin / "ggml/src/ggml-cuda/CMakeLists.txt").write_text("# fixture\n")
        bp.git(self.origin, ["add", "--", "."])
        self.commit(self.origin, "baseline")
        self.base = bp.git(self.origin, ["rev-parse", "HEAD"]).stdout.decode().strip()
        (self.origin / "src/target.cpp").write_bytes(self.after)
        self.patch = self.patches / "0001.patch"
        self.patch.write_bytes(bp.diff_bytes(self.origin, ["src/target.cpp"]))
        (self.origin / "src/target.cpp").write_bytes(self.before)
        self.manifest = {
            "schema_version": 1, "release_id": "fixture-p01",
            "upstream": {"base_commit": self.base, "repository": "local fixture"},
            "patches": [{"id": "P01", "file": self.patch.name,
                         "sha256": bp.sha256_file(self.patch),
                         "targets": [{"path": "src/target.cpp", "before_sha256": digest(self.before),
                                      "after_sha256": digest(self.after)}]}],
        }
        self.manifest_path = self.kit / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        for destination in (self.baseline, self.candidate):
            subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
                            "-c", "protocol.file.allow=always",
                            "clone", "--quiet", "--local", "--no-hardlinks",
                            str(self.origin), str(destination)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=bp.git_env())
        (self.candidate / "src/target.cpp").write_bytes(self.after)
        self.write_receipts()
        self.build_root = self.root / "NEW build ; literal"

    def commit(self, repo, message):
        env = bp.git_env()
        env.update({"GIT_AUTHOR_DATE": "2026-11-01T00:00:00Z",
                    "GIT_COMMITTER_DATE": "2026-11-01T00:00:00Z"})
        subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
                        "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "-c", "commit.gpgsign=false", "-C", str(repo), "commit", "--quiet", "-m", message],
                       check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def write_receipts(self, new=False):
        for role, repo in (("baseline", self.baseline), ("candidate", self.candidate)):
            head = bp.git(repo, ["rev-parse", "HEAD"]).stdout.decode().strip()
            receipt = {
                "schema_version": 1, "role": role, "base_commit": head,
                "release_id": self.manifest["release_id"],
                "manifest_sha256": bp.sha256_file(self.manifest_path),
                "patchset_sha256": bp.patchset_hash(self.manifest),
                "files": [{"path": "src/target.cpp", "sha256": bp.sha256_file(repo / "src/target.cpp")}],
                "status": "UNVALIDATED_NEW_BASE" if new else
                          ("BASELINE_READY" if role == "baseline" else "PATCHED_BASELINE"),
                "application_state": "SKIPPED_BASELINE" if role == "baseline" else "APPLIED",
                "validation_status": "UNVALIDATED",
                "diff_sha256": digest(bp.diff_bytes(repo, ["src/target.cpp"])),
            }
            (repo / bp.RECEIPT).write_text(json.dumps(receipt))

    def args(self, *extra):
        return ["--baseline", str(self.baseline), "--candidate", str(self.candidate),
                "--build-root", str(self.build_root), "--manifest", str(self.manifest_path),
                "--jobs", "1", *extra]

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = bp.main(self.args(*extra))
        return code, out.getvalue(), err.getvalue()

    def test_default_dry_run_never_creates_build_root_or_runs_cmake(self):
        with mock.patch.object(bp, "run_cmake_step", side_effect=AssertionError("CMake must not run")):
            code, out, err = self.run_main()
        self.assertEqual(code, 0, err)
        plan = json.loads(out)
        self.assertEqual(plan["mode"], "dry-run")
        self.assertFalse(self.build_root.exists())
        self.assertFalse(plan["inference_executed"])
        self.assertEqual([s["phase"] for s in plan["steps"]], ["configure", "build", "configure", "build"])
        self.assertIn("-DGGML_CUDA=OFF", plan["steps"][0]["command"])
        self.assertIn("-DLLAMA_USE_PREBUILT_UI=OFF", plan["steps"][0]["command"])
        self.assertFalse(any("LLAMA_CURL" in a for s in plan["steps"] for a in s["command"]))
        self.assertEqual((self.baseline / "src/target.cpp").read_bytes(), self.before)
        self.assertEqual((self.candidate / "src/target.cpp").read_bytes(), self.after)

    def test_dirty_baseline_and_extra_candidate_change_are_rejected(self):
        for repo in (self.baseline, self.candidate):
            with self.subTest(repo=repo.name):
                (repo / "README").write_text("unapproved change\n")
                code, _, _ = self.run_main()
                self.assertEqual(code, 2)
                self.assertFalse(self.build_root.exists())
                (repo / "README").write_text("fixture\n")

    def test_target_and_receipt_tampering_cannot_replace_expected_hash(self):
        (self.candidate / "src/target.cpp").write_bytes(self.after + b"// extra mutation\n")
        self.write_receipts()
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("independent expectation", err)
        self.assertFalse(self.build_root.exists())

    def test_patch_tampering_and_receipt_role_are_rejected(self):
        original = self.patch.read_bytes()
        self.patch.write_bytes(original + b"\n")
        self.assertEqual(self.run_main()[0], 2)
        self.patch.write_bytes(original)
        receipt_path = self.candidate / bp.RECEIPT
        receipt = json.loads(receipt_path.read_text())
        receipt["role"] = "baseline"
        receipt_path.write_text(json.dumps(receipt))
        self.assertEqual(self.run_main()[0], 2)
        self.assertFalse(self.build_root.exists())

    def test_same_head_is_required(self):
        (self.candidate / "src/target.cpp").write_bytes(self.before)
        (self.candidate / "README").write_text("different head\n")
        bp.git(self.candidate, ["add", "--", "README"])
        self.commit(self.candidate, "different")
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("HEAD differ", err)

    def test_existing_and_source_nested_build_roots_are_refused(self):
        self.build_root.mkdir()
        marker = self.build_root / "keep"
        marker.write_text("untouched")
        self.assertEqual(self.run_main()[0], 2)
        self.assertEqual(marker.read_text(), "untouched")
        self.build_root = self.candidate / "build"
        self.assertEqual(self.run_main()[0], 2)
        self.assertFalse(self.build_root.exists())

    def test_untracked_file_and_mode_change_are_refused(self):
        untracked = self.candidate / "CMakeUserPresets.json"
        untracked.write_text("{}")
        self.assertEqual(self.run_main()[0], 2)
        untracked.unlink()
        target = self.candidate / "src/target.cpp"
        target.chmod(0o755)
        self.write_receipts()
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("mode changes", err)

    def test_local_clean_filter_is_rejected_before_it_can_modify_source(self):
        marker = self.baseline / "filter-executed"
        helper = self.root / "clean helper.py"
        helper.write_text("import pathlib, sys\npathlib.Path(" + repr(str(marker)) +
                          ").write_text('unexpected write')\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        command = shlex.quote(sys.executable) + " " + shlex.quote(str(helper))
        bp.git(self.baseline, ["config", "filter.fixture.clean", command])
        (self.baseline / ".git/info/attributes").write_text("src/target.cpp filter=fixture\n")
        (self.baseline / "src/target.cpp").write_bytes(self.before + b"// force worktree inspection\n")
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("Local Git filters", err)
        self.assertFalse(marker.exists())
        self.assertFalse(self.build_root.exists())

    def test_gitlink_is_rejected_before_recursive_child_filter(self):
        child = self.baseline / "vendor/child"
        child.parent.mkdir()
        subprocess.run(["git", "-c", "protocol.file.allow=always", "clone", "--quiet", "--local",
                        "--no-hardlinks", str(self.origin), str(child)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=bp.git_env())
        bp.git(self.baseline, ["add", "--", "vendor/child"])
        self.commit(self.baseline, "fixture gitlink")
        marker = self.baseline / "child-filter-executed"
        helper = self.root / "child clean helper.py"
        helper.write_text("import pathlib, sys\npathlib.Path(" + repr(str(marker)) +
                          ").write_text('unexpected write')\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        bp.git(child, ["config", "filter.fixture.clean",
                       shlex.quote(sys.executable) + " " + shlex.quote(str(helper))])
        (child / ".git/info/attributes").write_text("src/target.cpp filter=fixture\n")
        (child / "src/target.cpp").write_bytes(self.before + b"// inspect child\n")
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("Gitlinks/submodules", err)
        self.assertFalse(marker.exists())
        self.assertFalse(self.build_root.exists())

    def test_unsafe_patch_path_is_refused(self):
        self.manifest["patches"][0]["file"] = "../outside.patch"
        self.manifest_path.write_text(json.dumps(self.manifest))
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("Unsafe manifest path", err)

    def test_cuda_is_explicit_and_architecture_cannot_inject_cmake_options(self):
        self.assertEqual(self.run_main("--cuda-arch", "120")[0], 2)
        self.assertEqual(self.run_main("--cuda", "--cuda-arch", "120;-Pbad.cmake")[0], 2)
        code, out, err = self.run_main("--cuda", "--cuda-arch", "120")
        self.assertEqual(code, 0, err)
        command = json.loads(out)["steps"][0]["command"]
        self.assertIn("-DGGML_CUDA=ON", command)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", command)

    def make_new_base(self, already_applied=False):
        prefix = b"// a fresh upstream context\n"
        for repo in (self.baseline, self.candidate):
            (repo / "src/target.cpp").write_bytes(prefix + (self.after if already_applied else self.before))
            bp.git(repo, ["add", "--", "src/target.cpp"])
            self.commit(repo, "new upstream")
        if not already_applied:
            (self.candidate / "src/target.cpp").write_bytes(prefix + self.after)
        self.write_receipts(new=True)

    def test_new_base_requires_opt_in_and_independent_reconstruction(self):
        self.make_new_base()
        initial = (self.baseline / "src/target.cpp").read_bytes()
        self.assertEqual(self.run_main()[0], 2)
        code, out, err = self.run_main("--allow-unvalidated-base")
        self.assertEqual(code, 0, err)
        verify = json.loads(out)["verification"]
        self.assertEqual(verify["status"], "BUILD_ONLY_NEW_BASE")
        self.assertEqual(verify["semantic_validation"], "NOT_RUN")
        self.assertEqual(verify["independent_new_base_patch_reconstruction"][0]["state"], "APPLIED")
        self.assertEqual((self.baseline / "src/target.cpp").read_bytes(), initial)
        self.assertFalse(self.build_root.exists())
        (self.candidate / "src/target.cpp").write_bytes(initial + b"// forged\n")
        self.write_receipts(new=True)
        self.assertEqual(self.run_main("--allow-unvalidated-base")[0], 2)

    def test_already_present_on_new_base_is_detected_without_claiming_speedup(self):
        self.make_new_base(already_applied=True)
        code, out, err = self.run_main("--allow-unvalidated-base")
        self.assertEqual(code, 0, err)
        verify = json.loads(out)["verification"]
        self.assertEqual(verify["changed_target_files"], [])
        self.assertEqual(verify["independent_new_base_patch_reconstruction"][0]["state"], "ALREADY_PRESENT")

    def overlay_pair(self, new=False):
        spec = importlib.util.spec_from_file_location("overlay_for_pair_test", MODULE.with_name("overlay.py"))
        overlay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(overlay)
        ref = None
        if new:
            (self.origin / "src/target.cpp").write_bytes(b"// fresh upstream context\n" + self.before)
            bp.git(self.origin, ["add", "--", "src/target.cpp"])
            self.commit(self.origin, "fresh upstream")
            ref = bp.git(self.origin, ["rev-parse", "HEAD"]).stdout.decode().strip()
        self.baseline = self.root / "real overlay baseline"
        self.candidate = self.root / "real overlay candidate"
        overlay.prepare(self.origin, self.baseline, self.manifest_path, self.patches, ref, True)
        overlay.prepare(self.origin, self.candidate, self.manifest_path, self.patches, ref, False)

    def test_receipts_created_by_real_overlay_are_accepted(self):
        self.overlay_pair()
        code, _, err = self.run_main()
        self.assertEqual(code, 0, err)
        self.assertFalse(self.build_root.exists())

    def test_new_base_real_overlay_is_independently_verified(self):
        self.overlay_pair(new=True)
        code, out, err = self.run_main("--allow-unvalidated-base")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["verification"]["status"], "BUILD_ONLY_NEW_BASE")
        self.assertFalse(self.build_root.exists())

    def fake_cmake(self, fail=False):
        path = self.root / "fake cmake"
        # Creates inert stand-ins only. If the harness ran one, it would leave a
        # sentinel; no inference executable or compiler is invoked by the test.
        source = '''import pathlib, sys
args=sys.argv[1:]
if "-S" in args:
    dest=pathlib.Path(args[args.index("-B")+1]); dest.mkdir(parents=True)
    if FAIL: sys.exit(7)
    (dest/"CMakeCache.txt").write_text("CMAKE_CXX_COMPILER:FILEPATH=/fixture/compiler\\nCMAKE_GENERATOR:INTERNAL=Fixture\\n")
    c=dest/"CMakeFiles"/"1.0"; c.mkdir(parents=True)
    (c/"CMakeCXXCompiler.cmake").write_text('set(CMAKE_CXX_COMPILER_ID "Fixture")\\nset(CMAKE_CXX_COMPILER_VERSION "1.2.3")\\n')
    phase="configure"
else:
    dest=pathlib.Path(args[args.index("--build")+1]); b=dest/"bin"; b.mkdir()
    for name in ("llama-server","llama-cli","llama-bench"):
        (b/name).write_text("DO NOT EXECUTE: fixture-only tool")
    (b/"libllama.so").write_bytes(b"fixture library")
    phase="build"
with (dest.parent/"order.txt").open("a") as h: h.write(dest.name+":"+phase+"\\n")
print("fixture CMake",phase)
'''
        path.write_text("#!" + sys.executable + "\nFAIL=" + repr(fail) + "\n" + source)
        path.chmod(0o755)
        return path

    def test_execute_is_sequential_and_records_hashes_without_running_tools(self):
        cmake = self.fake_cmake()
        code, out, err = self.run_main("--execute", "--cmake", str(cmake))
        self.assertEqual(code, 0, err)
        record = json.loads((self.build_root / "build_record.json").read_text())
        self.assertEqual(record["status"], "BUILD_ONLY_KNOWN_BASE")
        self.assertFalse(record["inference_executed"])
        self.assertFalse(record["tests_executed"])
        self.assertEqual(len(record["commands"]), 4)
        self.assertEqual((self.build_root / "order.txt").read_text().splitlines(),
                         ["baseline:configure", "baseline:build", "candidate:configure", "candidate:build"])
        for role in ("baseline", "candidate"):
            self.assertEqual(record["builds"][role]["compiler"]["CMAKE_CXX_COMPILER_VERSION"], "1.2.3")
            self.assertEqual(len(record["builds"][role]["artifacts"]), 4)
            self.assertTrue(all(len(x["sha256"]) == 64 for x in record["builds"][role]["artifacts"]))
        self.assertEqual(json.loads(out)["exit_code"], 0)

    def test_failure_stops_before_candidate_and_keeps_record(self):
        cmake = self.fake_cmake(fail=True)
        code, _, _ = self.run_main("--execute", "--cmake", str(cmake))
        self.assertEqual(code, 7)
        record = json.loads((self.build_root / "build_record.json").read_text())
        self.assertEqual(record["status"], "BUILD_FAILED")
        self.assertEqual(len(record["commands"]), 1)
        self.assertFalse((self.build_root / "candidate").exists())
        self.assertFalse(record["inference_executed"])


if __name__ == "__main__":
    unittest.main()
