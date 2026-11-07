"""Local-only trust-boundary tests for the portable overlay CLI.

These create tiny disposable Git repositories. They do not build llama.cpp,
load models, contact a network, or claim hardware/performance validation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "overlay.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(shutil.which("git"), "Git is required for local fixture tests")
class OverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="llama-overlay-test-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "upstream with spaces"
        self.source.mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                         "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
        self.git(self.source, "init", "-q")
        self.git(self.source, "config", "user.name", "Overlay fixture")
        self.git(self.source, "config", "user.email", "fixture@example.invalid")
        (self.source / "src").mkdir()
        self.target = self.source / "src" / "cache.cpp"
        self.before = ("// fixture prefix\n" * 10 +
                       "int cache_value() {\n    return 1;\n}\n" +
                       "// fixture suffix\n" * 10).encode()
        self.after = self.before.replace(b"return 1;", b"return 2;")
        self.target.write_bytes(self.before)
        (self.source / "README").write_text("Local fixture only.\n")
        self.git(self.source, "add", "--all")
        self.git(self.source, "commit", "-qm", "fixture base")
        self.base = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        self.target.write_bytes(self.after)
        self.patch = self.git(self.source, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD")
        self.target.write_bytes(self.before)
        self.kit = self.root / "kit"
        self.patches = self.kit / "patches"
        self.patches.mkdir(parents=True)
        self.patch_path = self.patches / "0001-observe.patch"
        self.patch_path.write_bytes(self.patch)
        self.manifest_path = self.kit / "manifest.json"
        self.manifest = {
            "schema_version": 1,
            "release_id": "fixture-1",
            "upstream": {"repository": "https://example.invalid/no-network", "base_commit": self.base},
            "patches": [{"id": "observe", "file": self.patch_path.name,
                         "sha256": sha(self.patch),
                         "targets": [{"path": "src/cache.cpp", "before_sha256": sha(self.before),
                                      "after_sha256": sha(self.after)}]}],
        }
        self.save_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, cwd: Path, *args: str) -> bytes:
        result = subprocess.run(["git", "-c", f"core.hooksPath={os.devnull}",
                                 "-c", "core.autocrlf=false", *args], cwd=cwd,
                                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        if result.returncode:
            self.fail(result.stderr.decode("utf-8", "replace"))
        return result.stdout

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")

    def source_snapshot(self) -> dict:
        return {
            "head": (self.source / ".git" / "HEAD").read_bytes(),
            "head_commit": self.git(self.source, "rev-parse", "HEAD"),
            "index": (self.source / ".git" / "index").read_bytes(),
            "files": {str(p.relative_to(self.source)): p.read_bytes()
                      for p in self.source.rglob("*")
                      if p.is_file() and ".git" not in p.relative_to(self.source).parts},
        }

    def invoke(self, operation: str, *args: str, expected_code: int = 0) -> dict:
        result = subprocess.run([sys.executable, str(TOOL), operation, "--source", str(self.source),
                                 "--manifest", str(self.manifest_path), *args],
                                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        self.assertEqual(result.returncode, expected_code,
                         result.stdout.decode() + result.stderr.decode())
        self.assertEqual(result.stderr, b"")
        return json.loads(result.stdout)

    def test_inspect_and_prepare_preserve_source_and_apply_exactly(self) -> None:
        # An untracked user file must not be removed or copied into the committed overlay.
        (self.source / "user-notes.txt").write_text("Keep this unchanged.\n")
        saved = self.source_snapshot()
        inspection = self.invoke("inspect")
        self.assertEqual(inspection["tree_state"], "BASELINE")
        self.assertEqual(inspection["patch_checks"][0]["status"], "CHECK_PASSED")
        self.assertEqual(self.source_snapshot(), saved)
        destination = self.root / "candidate with spaces"
        result = self.invoke("prepare", "--destination", str(destination))
        self.assertEqual(result["status"], "PATCHED_BASELINE")
        self.assertEqual(result["application_state"], "APPLIED")
        self.assertEqual((destination / "src/cache.cpp").read_bytes(), self.after)
        self.assertEqual(self.git(destination, "rev-parse", "HEAD").decode().strip(), self.base)
        self.assertFalse((destination / "user-notes.txt").exists())
        self.assertEqual(self.source_snapshot(), saved)
        receipt = json.loads((destination / ".llama-overlay-receipt.json").read_text())
        self.assertEqual(receipt["role"], "candidate")
        self.assertEqual(receipt["files"], [{"path": "src/cache.cpp", "sha256": sha(self.after)}])
        self.assertEqual(receipt["manifest_sha256"], sha(self.manifest_path.read_bytes()))
        canonical = json.dumps(self.manifest["patches"], sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(receipt["patchset_sha256"], sha(canonical))
        diff = self.git(destination, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--", "src/cache.cpp")
        self.assertEqual(receipt["diff_sha256"], sha(diff))
        self.assertEqual(receipt["validation_status"], "UNVALIDATED")
        # Independent objects: the clone must not depend on source alternates or hardlinks.
        self.assertFalse((destination / ".git/objects/info/alternates").exists())
        for source_object in (self.source / ".git/objects").glob("??/*"):
            copied = destination / ".git/objects" / source_object.relative_to(self.source / ".git/objects")
            if copied.exists():
                self.assertNotEqual((source_object.stat().st_dev, source_object.stat().st_ino),
                                    (copied.stat().st_dev, copied.stat().st_ino))

    def test_baseline_only_creates_matching_unpatched_receipt(self) -> None:
        destination = self.root / "baseline"
        saved = self.source_snapshot()
        result = self.invoke("prepare", "--destination", str(destination), "--baseline-only")
        self.assertEqual(result["status"], "BASELINE_READY")
        self.assertEqual(result["role"], "baseline")
        self.assertEqual(result["files"], [{"path": "src/cache.cpp", "sha256": sha(self.before)}])
        self.assertEqual(self.git(destination, "diff", "HEAD"), b"")
        self.assertEqual(self.source_snapshot(), saved)

    def test_existing_destination_is_not_overwritten(self) -> None:
        destination = self.root / "exists"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_bytes(b"owned by user")
        saved = self.source_snapshot()
        result = self.invoke("prepare", "--destination", str(destination), expected_code=2)
        self.assertEqual(result["code"], "DESTINATION_EXISTS")
        self.assertEqual(marker.read_bytes(), b"owned by user")
        self.assertEqual(list(destination.iterdir()), [marker])
        self.assertEqual(self.source_snapshot(), saved)

    def test_patch_hash_mismatch_stops_before_clone(self) -> None:
        self.patch_path.write_bytes(self.patch + b"\nmodified after release\n")
        destination = self.root / "candidate"
        saved = self.source_snapshot()
        result = self.invoke("prepare", "--destination", str(destination), expected_code=2)
        self.assertEqual(result["code"], "PATCH_HASH_MISMATCH")
        self.assertFalse(destination.exists())
        self.assertEqual(self.source_snapshot(), saved)

    def test_trusted_after_hash_is_enforced_on_pinned_base(self) -> None:
        self.manifest["patches"][0]["targets"][0]["after_sha256"] = "a" * 64
        self.save_manifest()
        destination = self.root / "candidate"
        saved = self.source_snapshot()
        result = self.invoke("prepare", "--destination", str(destination), expected_code=2)
        self.assertEqual(result["status"], "INCOMPATIBLE")
        self.assertEqual(result["code"], "AFTER_HASH_MISMATCH")
        failure = json.loads((destination / ".llama-overlay-receipt.json").read_text())
        self.assertEqual(failure["status"], "INCOMPATIBLE")
        self.assertEqual(self.source_snapshot(), saved)

    def test_new_base_requires_explicit_full_commit_and_preserves_other_changes(self) -> None:
        # Outside the patch context, but within the same file: before/after full hashes differ.
        extension = b"// upstream extension outside reviewed hunk\n"
        self.target.write_bytes(self.before + extension)
        self.git(self.source, "add", "--all")
        self.git(self.source, "commit", "-qm", "new upstream")
        new_commit = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        saved = self.source_snapshot()
        rejected = self.root / "implicit-new-base"
        result = self.invoke("prepare", "--destination", str(rejected), expected_code=2)
        self.assertEqual(result["code"], "EXPLICIT_REF_REQUIRED")
        self.assertFalse(rejected.exists())
        short = self.invoke("prepare", "--destination", str(rejected), "--ref", new_commit[:12], expected_code=2)
        self.assertEqual(short["code"], "REF_NOT_FULL_SHA")
        destination = self.root / "explicit-new-base"
        result = self.invoke("prepare", "--destination", str(destination), "--ref", new_commit)
        self.assertEqual(result["status"], "UNVALIDATED_NEW_BASE")
        self.assertEqual(result["base_commit"], new_commit)
        self.assertEqual((destination / "src/cache.cpp").read_bytes(), self.after + extension)
        self.assertEqual(result["files"][0]["sha256"], sha(self.after + extension))
        baseline = self.invoke("prepare", "--destination", str(self.root / "new-baseline"),
                               "--ref", new_commit, "--baseline-only")
        self.assertEqual(baseline["role"], "baseline")
        self.assertEqual(baseline["status"], "UNVALIDATED_NEW_BASE")
        self.assertEqual(baseline["files"][0]["sha256"], sha(self.before + extension))
        self.assertEqual(self.source_snapshot(), saved)

    def test_local_replace_refs_do_not_reinterpret_the_selected_commit(self) -> None:
        self.target.write_bytes(self.before.replace(b"return 1;", b"return 99;"))
        self.git(self.source, "add", "--all")
        self.git(self.source, "commit", "-qm", "replacement object")
        replacement = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        self.git(self.source, "checkout", "--detach", self.base)
        self.git(self.source, "replace", self.base, replacement)
        saved = self.source_snapshot()
        result = self.invoke("prepare", "--destination", str(self.root / "candidate"))
        self.assertEqual(result["base_commit"], self.base)
        self.assertEqual((self.root / "candidate/src/cache.cpp").read_bytes(), self.after)
        self.assertEqual(self.source_snapshot(), saved)

    def test_context_conflict_is_machine_readable_without_source_changes(self) -> None:
        self.target.write_bytes(self.before.replace(b"return 1;", b"return 99;"))
        self.git(self.source, "add", "--all")
        self.git(self.source, "commit", "-qm", "conflicting upstream")
        new_commit = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        saved = self.source_snapshot()
        destination = self.root / "conflict"
        result = self.invoke("prepare", "--destination", str(destination), "--ref", new_commit,
                             expected_code=2)
        self.assertEqual(result["status"], "INCOMPATIBLE")
        self.assertEqual(result["code"], "CONTEXT_CONFLICT")
        self.assertEqual(self.git(destination, "diff", "HEAD"), b"")
        self.assertEqual(self.source_snapshot(), saved)

    def test_fully_patched_tree_is_detected_and_not_applied_twice(self) -> None:
        self.target.write_bytes(self.after)
        self.git(self.source, "add", "--all")
        self.git(self.source, "commit", "-qm", "upstream already has reviewed bytes")
        new_commit = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        saved = self.source_snapshot()
        inspection = self.invoke("inspect")
        self.assertEqual(inspection["tree_state"], "PATCHED")
        destination = self.root / "already-patched"
        result = self.invoke("prepare", "--destination", str(destination), "--ref", new_commit)
        self.assertEqual(result["status"], "UNVALIDATED_NEW_BASE")
        self.assertEqual(result["application_state"], "ALREADY_PRESENT")
        self.assertEqual(result["applied_patch_ids"], [])
        self.assertEqual(self.git(destination, "diff", "HEAD"), b"")
        self.assertEqual(self.source_snapshot(), saved)

    def test_dirty_source_is_rejected_without_resetting_it(self) -> None:
        self.target.write_bytes(self.before + b"// user work in progress\n")
        saved = self.source_snapshot()
        destination = self.root / "candidate"
        result = self.invoke("prepare", "--destination", str(destination), expected_code=2)
        self.assertEqual(result["code"], "SOURCE_DIRTY")
        self.assertFalse(destination.exists())
        self.assertEqual(self.source_snapshot(), saved)

    def test_git_clean_filter_cannot_execute_during_inspection_or_prepare(self) -> None:
        # Arm the exact failure mode: Git status would run a clean filter when
        # the tracked file has changed. The tool must refuse before that read.
        (self.source / ".gitattributes").write_text("src/cache.cpp filter=readonly_probe\n")
        self.git(self.source, "add", ".gitattributes")
        self.git(self.source, "commit", "-qm", "declare filter attribute")
        commit = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        marker = self.source / "FILTER_EXECUTED"
        helper = self.root / "filter-helper.py"
        helper.write_text("from pathlib import Path\nimport sys\n"
                          f"Path({str(marker)!r}).write_text('unexpected mutation')\n"
                          "sys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        self.git(self.source, "config", "filter.readonly_probe.clean",
                 shlex.quote(sys.executable) + " " + shlex.quote(str(helper)))
        self.target.write_bytes(self.before + b"// dirty content triggers a clean filter\n")
        saved = self.source_snapshot()
        result = self.invoke("inspect", expected_code=2)
        self.assertEqual(result["code"], "UNSAFE_GIT_FILTER")
        self.assertFalse(marker.exists())
        destination = self.root / "candidate"
        result = self.invoke("prepare", "--destination", str(destination), "--ref", commit,
                             expected_code=2)
        self.assertEqual(result["code"], "UNSAFE_GIT_FILTER")
        self.assertFalse(marker.exists())
        self.assertFalse(destination.exists())
        self.assertEqual(self.source_snapshot(), saved)

    def test_gitlink_is_rejected_before_recursive_filter_inspection(self) -> None:
        nested = self.source / "vendor"
        nested.mkdir()
        self.git(nested, "init", "-q")
        self.git(nested, "config", "user.name", "Nested fixture")
        self.git(nested, "config", "user.email", "nested@example.invalid")
        nested_target = nested / "probe.txt"
        nested_target.write_text("nested tracked content\n")
        (nested / ".gitattributes").write_text("probe.txt filter=nested_probe\n")
        self.git(nested, "add", "--all")
        self.git(nested, "commit", "-qm", "nested base")
        nested_commit = self.git(nested, "rev-parse", "HEAD").decode().strip()
        self.git(self.source, "update-index", "--add", "--cacheinfo",
                 f"160000,{nested_commit},vendor")
        self.git(self.source, "commit", "-qm", "record gitlink")
        commit = self.git(self.source, "rev-parse", "HEAD").decode().strip()
        marker = self.source / "SUBMODULE_FILTER_EXECUTED"
        helper = self.root / "nested-filter-helper.py"
        helper.write_text("from pathlib import Path\nimport sys\n"
                          f"Path({str(marker)!r}).write_text('unexpected recursive filter')\n"
                          "sys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        self.git(nested, "config", "filter.nested_probe.clean",
                 shlex.quote(sys.executable) + " " + shlex.quote(str(helper)))
        nested_target.write_text("modified nested content that would invoke its filter\n")
        saved = self.source_snapshot()
        nested_index = (nested / ".git/index").read_bytes()
        for operation in ("inspect", "prepare"):
            extra = [] if operation == "inspect" else ["--destination", str(self.root / "candidate"),
                                                       "--ref", commit]
            result = self.invoke(operation, *extra, expected_code=2)
            self.assertEqual(result["code"], "UNSUPPORTED_GITLINK")
            self.assertFalse(marker.exists())
            self.assertEqual((nested / ".git/index").read_bytes(), nested_index)
            self.assertEqual(self.source_snapshot(), saved)
        self.assertFalse((self.root / "candidate").exists())

    def test_destination_inside_source_is_rejected(self) -> None:
        saved = self.source_snapshot()
        destination = self.source / "new-overlay"
        result = self.invoke("prepare", "--destination", str(destination), expected_code=2)
        self.assertEqual(result["code"], "UNSAFE_DESTINATION")
        self.assertFalse(destination.exists())
        self.assertEqual(self.source_snapshot(), saved)

    def test_manifest_traversal_and_undeclared_patch_target_are_rejected(self) -> None:
        self.manifest["patches"][0]["file"] = "../outside.patch"
        self.save_manifest()
        result = self.invoke("inspect", expected_code=2)
        self.assertEqual(result["code"], "INVALID_PATH")
        self.manifest["patches"][0]["file"] = self.patch_path.name
        self.manifest["patches"][0]["targets"][0]["path"] = "README"
        self.save_manifest()
        result = self.invoke("inspect", expected_code=2)
        self.assertEqual(result["code"], "PATCH_TARGET_MISMATCH")

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlinks unavailable")
    def test_symlink_patch_file_is_rejected(self) -> None:
        external = self.root / "external.patch"
        external.write_bytes(self.patch)
        self.patch_path.unlink()
        try:
            self.patch_path.symlink_to(external)
        except OSError:
            self.skipTest("Symlink creation not permitted")
        result = self.invoke("inspect", expected_code=2)
        self.assertEqual(result["code"], "UNSAFE_FILE")


if __name__ == "__main__":
    unittest.main()
