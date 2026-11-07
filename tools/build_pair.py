#!/usr/bin/env python3
"""Validate and optionally compile a baseline/candidate pair; never run inference.

The default is a read-only plan (apart from an automatically removed temporary
patch reconstruction). --execute creates a NEW build root, then runs CMake
configure/build sequentially. Neither checkout is prepared or changed here.
Compiler feature probes performed by CMake are not model inference tests.

Project options were checked against llama.cpp dde879bc4f954b809892fd0d660515f42536394a:
  CMakeLists.txt, ggml/CMakeLists.txt, tools/ui/CMakeLists.txt,
  ggml/src/ggml-cuda/CMakeLists.txt. LLAMA_CURL is deprecated and is not used.
These switches disable the known dependency/UI download paths, not arbitrary
network access by an unreviewed future CMake project or compiler wrapper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

RECEIPT = ".llama-overlay-receipt.json"
TARGETS = ("llama-server", "llama-cli", "llama-bench")
MAX_JSON = 4 * 1024 * 1024
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

# Only explicit, reviewed options. No caller-supplied -D, -P, install target,
# toolchain/preset, arbitrary build target, shell fragment or download command.
PROJECT_OPTIONS = {
    "LLAMA_BUILD_COMMON": "ON", "LLAMA_BUILD_TOOLS": "ON",
    "LLAMA_BUILD_SERVER": "ON", "LLAMA_BUILD_TESTS": "OFF",
    "LLAMA_BUILD_EXAMPLES": "OFF", "LLAMA_BUILD_APP": "OFF",
    "LLAMA_BUILD_UI": "OFF", "LLAMA_USE_PREBUILT_UI": "OFF",
    "LLAMA_CONNECT": "OFF", "LLAMA_LLGUIDANCE": "OFF",
    "LLAMA_USE_SYSTEM_GGML": "OFF",
    "GGML_CPU": "ON", "GGML_METAL": "OFF", "GGML_CCACHE": "OFF",
    "GGML_CPU_KLEIDIAI": "OFF", "GGML_OPENMP_FETCH": "OFF",
    "GGML_CUDA_CCCL_VERSION": "", "GGML_CUDA_NCCL": "OFF",
}
DOWNLOAD_OPTIONS = {k: PROJECT_OPTIONS[k] for k in (
    "LLAMA_BUILD_UI", "LLAMA_USE_PREBUILT_UI", "LLAMA_CONNECT",
    "LLAMA_LLGUIDANCE", "GGML_CPU_KLEIDIAI", "GGML_OPENMP_FETCH",
    "GGML_CUDA_CCCL_VERSION")}


class BuildPairError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BuildPairError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def regular_file(path):
    require(not path.is_symlink(), f"Symlink file refused: {path}")
    try:
        st = path.stat()
    except OSError as exc:
        raise BuildPairError(f"Required file is unavailable: {path}") from exc
    require(stat.S_ISREG(st.st_mode), f"Regular file required: {path}")
    return path


def read_json(path):
    regular_file(path)
    require(path.stat().st_size <= MAX_JSON, f"JSON size limit exceeded: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise BuildPairError(f"Invalid JSON: {path}") from exc
    require(isinstance(result, dict), f"JSON object required: {path}")
    return result


def relpath(value):
    require(isinstance(value, str) and value and "\\" not in value and
            not any(ord(c) < 32 for c in value), "Unsafe manifest path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and
            all(x not in (".", "..", ".git") for x in p.parts),
            f"Unsafe manifest path: {value}")
    return value


def contained_file(root, value):
    value = relpath(value)
    p = root
    for component in PurePosixPath(value).parts:
        p = p / component
        require(not p.is_symlink(), f"Symlink in source path: {value}")
    regular_file(p)
    require(p.resolve().is_relative_to(root.resolve()), "File escapes its root")
    return p


def git_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull, "GIT_PROTOCOL_FROM_USER": "0",
                "GIT_NO_REPLACE_OBJECTS": "1", "LC_ALL": "C"})
    return env


def git(root, args, *, allowed=(0,)):
    exe = shutil.which("git")
    require(exe is not None, "git is required for validation")
    argv = [exe, "--no-optional-locks", "-c", "core.fsmonitor=false",
            "-c", "core.untrackedCache=false", "-c", "core.hooksPath=" + os.devnull,
            "-c", "core.attributesFile=" + os.devnull,
            "-C", str(root), *args]
    try:
        proc = subprocess.run(argv, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=False, env=git_env(), timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildPairError("Git validation could not complete") from exc
    require(proc.returncode in allowed, "Git validation failed: " + " ".join(args[:2]))
    require(len(proc.stdout) <= 64 * 1024 * 1024, "Git validation output limit exceeded")
    return proc


def patchset_hash(manifest):
    records = [{k: item[k] for k in ("id", "file", "sha256", "targets")}
               for item in manifest["patches"]]
    return hashlib.sha256(json.dumps(records, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def load_manifest(path, patches_dir):
    manifest = read_json(path)
    require(manifest.get("schema_version") == 1, "Unsupported manifest schema")
    require(COMMIT_RE.fullmatch(manifest.get("upstream", {}).get("base_commit", ""))
            is not None, "Manifest requires a full upstream.base_commit")
    require(isinstance(manifest.get("release_id"), str) and manifest["release_id"],
            "Manifest release_id is missing")
    patches = manifest.get("patches")
    require(isinstance(patches, list) and patches, "Manifest has no patches")
    before, after, ids = {}, {}, set()
    for patch in patches:
        require(isinstance(patch, dict), "Invalid patch entry")
        require(isinstance(patch.get("id"), str) and patch["id"] not in ids,
                "Missing or duplicate patch id")
        ids.add(patch["id"])
        require(HASH_RE.fullmatch(patch.get("sha256", "")) is not None,
                "Patch SHA256 is missing")
        source = contained_file(patches_dir, patch.get("file"))
        require(sha256_file(source) == patch["sha256"], "Patch content differs from manifest")
        targets = patch.get("targets")
        require(isinstance(targets, list) and targets, "Patch target list is empty")
        seen = set()
        for target in targets:
            require(isinstance(target, dict), "Patch target must be an object")
            name = relpath(target.get("path"))
            require(name not in seen, "Duplicate target in patch")
            seen.add(name)
            a, b = target.get("before_sha256", ""), target.get("after_sha256", "")
            require(HASH_RE.fullmatch(a) is not None and HASH_RE.fullmatch(b) is not None,
                    "Target before/after SHA256 is missing")
            if name in after:
                require(after[name] == a, "Patch hash chain is inconsistent")
            before.setdefault(name, a)
            after[name] = b
    extra = manifest.get("additional_files", [])
    require(isinstance(extra, list) and len(extra) <= 20, "Additional file list is malformed")
    seen_sources, seen_destinations = set(), set()
    for item in extra:
        require(isinstance(item, dict) and set(item) == {"source", "destination", "sha256"},
                "Malformed vendor source record")
        src, dst = relpath(item["source"]), relpath(item["destination"])
        vendor_ok = (src in ("modules/engram/row_cache.hpp", "modules/engram/mapped_rows.hpp",
                             "modules/engram/model_cache.hpp", "modules/engram/graph_stage.hpp") and
                     dst == "src/llama-experimental/" + src.rsplit("/", 1)[-1]) or (
                         src in ("modules/moe/upload_pipeline.hpp", "modules/moe/upload_types.hpp", "modules/moe/upload_worker.hpp", "modules/moe/upload_plan.hpp", "modules/moe/upload_scheduler.hpp", "modules/moe/upload_completion.hpp") and
                         dst == "src/llama-experimental/" + src.rsplit("/", 1)[-1]) or (
                         src in ("modules/cuda/hc_post_reuse.cuh", "modules/cuda/mmq_tile_cache.hpp", "modules/cuda/mmq_select_probe.hpp") and
                         dst == "ggml/src/ggml-cuda/llama-experimental/" + src.rsplit("/", 1)[-1]) or (
                         src == "modules/speculative/mtp_verify_window.hpp" and
                         dst == "common/llama-experimental/mtp_verify_window.hpp") or (
                         src == "modules/speculative/mtp_rewind_guard.hpp" and
                         dst == "common/llama-experimental/mtp_rewind_guard.hpp")
        require(vendor_ok, "Vendor file is outside the reviewed vendor namespace")
        require(src not in seen_sources and dst not in seen_destinations and dst not in before,
                "Vendor file path conflict")
        require(HASH_RE.fullmatch(item["sha256"]) is not None, "Vendor sha256 missing")
        require(sha256_file(contained_file(path.parent, src)) == item["sha256"],
                "Vendor file changed since manifest")
        seen_sources.add(src)
        seen_destinations.add(dst)
    return manifest, before, after


def source_identity(root, allowed_untracked=None):
    require(root.is_dir() and (root / ".git").exists(), "A checkout with .git is required")
    top = git(root, ["rev-parse", "--show-toplevel"]).stdout.decode().strip()
    require(Path(top).resolve() == root, "Source must be the checkout root")
    head = git(root, ["rev-parse", "HEAD"]).stdout.decode().strip()
    require(COMMIT_RE.fullmatch(head) is not None, "A full SHA1 HEAD is required")
    # status/diff can invoke a local clean/process filter even with hooks and
    # external diff disabled. Refuse configured helpers before inspecting the
    # worktree, without displaying a potentially sensitive command value.
    filters = git(root, ["config", "--null", "--get-regexp",
                         r"^filter\..*\.(clean|process|smudge)$"], allowed=(0, 1)).stdout
    require(all(not row.partition(b"\n")[2].strip() for row in filters.split(b"\0") if row),
            "Local Git filters are not allowed for read-only source validation")
    # A recursive submodule diff can invoke a child repository's clean/process
    # filter, which the root config check cannot see. This kit has no gitlinks;
    # reject them before any diff instead of running nested worktree inspection.
    entries = git(root, ["ls-files", "--stage", "-z"]).stdout.split(b"\0")
    require(not any(row.startswith(b"160000 ") for row in entries),
            "Gitlinks/submodules are unsupported for read-only source validation")
    # Assume-unchanged / skip-worktree flags can conceal modified tracked files.
    flags = git(root, ["ls-files", "-v", "-z"]).stdout.split(b"\0")
    require(all(not row or (not chr(row[0]).islower() and row[:1] != b"S")
                for row in flags), "Hidden tracked-file flags are not allowed")
    require(not git(root, ["diff", "--cached", "--name-only", "-z", "HEAD", "--"]).stdout,
            "Staged changes are not allowed")
    require(not git(root, ["diff", "--no-ext-diff", "--no-textconv", "--summary", "HEAD", "--"]).stdout,
            "Add/delete/rename/mode changes are not allowed")
    raw = git(root, ["diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z",
                     "--ignore-submodules=none", "HEAD", "--"]).stdout
    changed = sorted(x.decode("utf-8") for x in raw.split(b"\0") if x)
    untracked = git(root, ["ls-files", "--others", "-z"]).stdout
    names = {x.decode("utf-8") for x in untracked.split(b"\0") if x}
    require(names <= {RECEIPT} | set(allowed_untracked or ()),
            "Unexpected untracked files in prepared checkout")
    return head, changed


def target_hashes(root, names):
    return {name: sha256_file(contained_file(root, name)) for name in sorted(names)}


def diff_bytes(root, names):
    return git(root, ["-c", "core.quotePath=false", "-c", "color.ui=false",
                      "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false",
                      "diff", "--binary", "--no-ext-diff", "--no-textconv",
                      "HEAD", "--", *sorted(names)]).stdout


def validate_receipt(root, role, manifest, manifest_path, expected_commit, expected_files):
    receipt_path = root / RECEIPT
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == 1 and receipt.get("role") == role,
            f"Wrong {role} receipt role/schema")
    require(receipt.get("base_commit") == expected_commit,
            f"{role} receipt commit differs from HEAD")
    require(receipt.get("release_id") == manifest["release_id"], "Receipt release differs")
    require(receipt.get("manifest_sha256") == sha256_file(manifest_path),
            "Receipt manifest SHA256 differs")
    require(receipt.get("patchset_sha256") == patchset_hash(manifest),
            "Receipt patchset SHA256 differs")
    assets = manifest.get("additional_files", [])
    if assets:
        encoded = json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()
        require(receipt.get("assetset_sha256") == hashlib.sha256(encoded).hexdigest(),
                "Receipt vendor-set checksum differs")
        expected_assets = [] if role == "baseline" else [
            {"path": x["destination"], "sha256": x["sha256"]} for x in assets]
        require(receipt.get("additional_files") == expected_assets,
                "Receipt vendor files differ")
        for item in expected_assets:
            require(sha256_file(contained_file(root, item["path"])) == item["sha256"],
                    "Vendor source changed after preparation")
        if role == "baseline":
            require(not any(os.path.lexists(root / x["destination"]) for x in assets),
                    "Baseline must not contain experimental vendor headers")
    entries = receipt.get("files")
    require(isinstance(entries, list), "Receipt files missing")
    files = {}
    for entry in entries:
        require(isinstance(entry, dict), "Invalid receipt file")
        name = relpath(entry.get("path"))
        require(name not in files and HASH_RE.fullmatch(entry.get("sha256", "")) is not None,
                "Invalid or duplicate receipt hash")
        files[name] = entry["sha256"]
    require(files == expected_files, f"{role} receipt target hashes differ from independent expectation")
    require(target_hashes(root, files) == files, f"{role} source target content changed")
    allowed_status = ({"BASELINE_READY", "UNVALIDATED_NEW_BASE"} if role == "baseline" else
                      {"PATCHED_BASELINE", "UNVALIDATED_NEW_BASE"})
    require(receipt.get("status") in allowed_status, f"Unexpected {role} receipt status")
    actual_diff = hashlib.sha256(diff_bytes(root, files)).hexdigest()
    require(receipt.get("diff_sha256") == actual_diff, "Receipt diff SHA256 differs")
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path),
            "status": receipt["status"], "diff_sha256": actual_diff,
            "application_state": receipt.get("application_state"),
            "validation_status": receipt.get("validation_status")}


def reconstruct_new_base(baseline, commit, manifest, patches_dir, target_names):
    """Only the owned temp tree is written; no source index/worktree mutations."""
    original, expected, applications = {}, {}, []
    with tempfile.TemporaryDirectory(prefix="llama-pair-validate-") as tmp:
        temporary = Path(tmp)
        git(temporary, ["-c", "init.templateDir=", "init", "--quiet"])
        for name in sorted(target_names):
            item = git(baseline, ["ls-tree", "-z", commit, "--", name]).stdout
            require(item.startswith((b"100644 blob ", b"100755 blob ")),
                    "Only existing regular tracked patch targets are supported")
            data = git(baseline, ["show", commit + ":" + name]).stdout
            dest = temporary / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if item.startswith(b"100755"):
                dest.chmod(0o755)
            original[name] = hashlib.sha256(data).hexdigest()
        for patch in manifest["patches"]:
            source = contained_file(patches_dir, patch["file"])
            check = git(temporary, ["apply", "--check", "--whitespace=error-all", "--", str(source)], allowed=(0, 1))
            if check.returncode == 0:
                git(temporary, ["apply", "--whitespace=error-all", "--", str(source)])
                state = "APPLIED"
            else:
                reverse = git(temporary, ["apply", "--reverse", "--check", "--whitespace=error-all", "--", str(source)], allowed=(0, 1))
                require(reverse.returncode == 0, "Patch does not apply exactly to selected new base")
                state = "ALREADY_PRESENT"
            applications.append({"id": patch["id"], "state": state})
        actual_paths = {p.relative_to(temporary).as_posix()
                        for p in temporary.rglob("*") if p.is_file() and
                        not p.relative_to(temporary).parts[0] == ".git"}
        require(actual_paths == set(target_names), "Patch changed undeclared file set")
        expected = target_hashes(temporary, target_names)
    return original, expected, applications


def check_project_options(root, cuda):
    inputs = [root / "CMakeLists.txt", root / "ggml/CMakeLists.txt"]
    body = "\n".join(regular_file(p).read_text(encoding="utf-8") for p in inputs)
    for option in [*PROJECT_OPTIONS, "GGML_CUDA"]:
        require(re.search(r"\b(?:option|set)\s*\(\s*" + re.escape(option) + r"\b", body, re.I),
                f"Required CMake option no longer declared: {option}")
    if cuda:
        regular_file(root / "ggml/src/ggml-cuda/CMakeLists.txt")
    return {str(p.relative_to(root)): sha256_file(p) for p in inputs}


def validate_pair(baseline, candidate, manifest_path, patches_dir, *, allow_new_base=False):
    require(baseline != candidate and not baseline.is_relative_to(candidate) and
            not candidate.is_relative_to(baseline), "Source checkouts must be separate and nonnested")
    manifest, before, after = load_manifest(manifest_path, patches_dir)
    assets = manifest.get("additional_files", [])
    base_head, base_changes = source_identity(baseline)
    cand_head, cand_changes = source_identity(candidate, {x["destination"] for x in assets})
    require(base_head == cand_head, "Baseline and candidate HEAD differ")
    require(not base_changes, "Baseline has tracked changes")
    for extra in assets:
        # Native upstream contents are never overwritten by this kit.
        require(not git(baseline, ["ls-tree", "HEAD", "--", extra["destination"]]).stdout,
                "Selected upstream already contains a declared experimental vendor file")
    known = base_head == manifest["upstream"]["base_commit"]
    applications = []
    if not known:
        require(allow_new_base, "New base requires --allow-unvalidated-base, including for a plan")
        before, after, applications = reconstruct_new_base(
            baseline, base_head, manifest, patches_dir, before)
    expected_changes = sorted(name for name in before if before[name] != after[name])
    require(cand_changes == expected_changes, "Candidate tracked diff paths differ from trusted patchset")
    records = {
        "baseline": validate_receipt(baseline, "baseline", manifest, manifest_path, base_head, before),
        "candidate": validate_receipt(candidate, "candidate", manifest, manifest_path, base_head, after),
    }
    return {"base_commit": base_head, "release_id": manifest["release_id"],
            "manifest_sha256": sha256_file(manifest_path), "patchset_sha256": patchset_hash(manifest),
            "known_base": known, "status": "BUILD_ONLY_KNOWN_BASE" if known else "BUILD_ONLY_NEW_BASE",
            "semantic_validation": "NOT_RUN", "target_files": sorted(before),
            "changed_target_files": expected_changes, "receipts": records,
            "independent_new_base_patch_reconstruction": applications}


def available_jobs():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def make_plan(args):
    baseline, candidate = args.baseline.expanduser().resolve(), args.candidate.expanduser().resolve()
    build_root = args.build_root.expanduser().absolute()
    require(not build_root.exists() and not build_root.is_symlink(), "Build root must not exist")
    require(build_root.parent.is_dir(), "Build root parent must already exist")
    build_root = build_root.parent.resolve() / build_root.name
    require(not build_root.exists(), "Resolved build root already exists")
    for source in (baseline, candidate):
        require(not build_root.is_relative_to(source) and not source.is_relative_to(build_root),
                "Build root must be outside both source checkouts")
    require(1 <= args.jobs <= available_jobs(), "--jobs must be within available logical CPUs")
    require(args.cuda or args.cuda_arch is None, "--cuda-arch requires --cuda")
    if args.cuda_arch is not None:
        require(re.fullmatch(r"[0-9]{2,3}(?:a|f)?(?:-real|-virtual)?", args.cuda_arch) is not None,
                "One explicit CUDA architecture is required (for example 120)")
    require(args.step_timeout > 0, "--step-timeout must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    patches_dir = (args.patches_dir.expanduser().resolve() if args.patches_dir else
                   manifest_path.parent / "patches")
    verification = validate_pair(baseline, candidate, manifest_path, patches_dir,
                                 allow_new_base=args.allow_unvalidated_base)
    cmake = shutil.which(args.cmake)
    if args.execute:
        require(cmake is not None, "cmake was not found; no build directory created")
    cmake = str(Path(cmake).resolve()) if cmake else args.cmake
    options = {**PROJECT_OPTIONS, "GGML_CUDA": "ON" if args.cuda else "OFF"}
    common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
              "-DCMAKE_C_COMPILER_LAUNCHER=", "-DCMAKE_CXX_COMPILER_LAUNCHER=",
              "-DCMAKE_CUDA_COMPILER_LAUNCHER="]
    common.extend(f"-D{k}={v}" for k, v in options.items())
    if args.cuda_arch is not None:
        common.append("-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch)
    steps, sources = [], {}
    for role, source in (("baseline", baseline), ("candidate", candidate)):
        sources[role] = {"path": str(source), "cmake_input_sha256": check_project_options(source, args.cuda)}
        destination = build_root / role
        steps.append({"role": role, "phase": "configure",
                      "command": [cmake, "-S", str(source), "-B", str(destination), *common]})
        steps.append({"role": role, "phase": "build",
                      "command": [cmake, "--build", str(destination), "--config", "Release",
                                  "--parallel", str(args.jobs), "--target", *TARGETS]})
    return {"schema_version": 1, "mode": "execute" if args.execute else "dry-run",
            "build_root": str(build_root), "sources": sources, "verification": verification,
            "cuda_requested": args.cuda, "cuda_arch_requested": args.cuda_arch,
            "jobs": args.jobs, "cmake": cmake, "download_features_disabled": DOWNLOAD_OPTIONS,
            "network_isolation_guaranteed": False,
            "new_base_cmake_semantics_reviewed": verification["known_base"],
            "inference_executed": False, "tests_executed": False,
            "temporary_patch_reconstruction_only": not verification["known_base"], "steps": steps}


def atomic_json(path, obj):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def stop_child(proc):
    if proc.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def run_cmake_step(step, build_root, timeout):
    log_path = build_root / (step["role"] + "-" + step["phase"] + ".log")
    record = {**step, "log": log_path.name, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    started = time.monotonic()
    with open(log_path, "xb") as log:
        try:
            proc = subprocess.Popen(step["command"], cwd=build_root, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, shell=False,
                                    start_new_session=(os.name == "posix"), env=git_env())
        except OSError as exc:
            record.update(returncode=127, error=type(exc).__name__)
        else:
            try:
                record["returncode"] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_child(proc)
                record.update(returncode=124, error="step_timeout")
            except KeyboardInterrupt:
                stop_child(proc)
                record.update(returncode=130, error="interrupted")
    record["elapsed_seconds"] = time.monotonic() - started
    record["log_sha256"] = sha256_file(log_path)
    return record


def compiler_record(build):
    keys = {"CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER",
            "CMAKE_MAKE_PROGRAM", "CMAKE_GENERATOR", "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_CACHE_MAJOR_VERSION", "CMAKE_CACHE_MINOR_VERSION", "CMAKE_CACHE_PATCH_VERSION"}
    result = {}
    cache = build / "CMakeCache.txt"
    if cache.is_file() and not cache.is_symlink():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"([^:#=]+):[^=]*=(.*)$", line)
            if m and m[1] in keys:
                result[m[1]] = m[2]
    files = build / "CMakeFiles"
    if files.is_dir():
        for path in sorted(files.glob("*/CMake*Compiler.cmake")):
            if path.is_symlink() or not path.resolve().is_relative_to(build):
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(r'set\((CMAKE_(?:C|CXX|CUDA)_COMPILER(?:_ID|_VERSION)?)\s+"([^"\r\n]*)"\)', line)
                if m:
                    result[m[1]] = m[2]
    return result


def build_artifacts(build):
    result = []
    if not build.is_dir():
        return result
    for path in sorted(build.rglob("*")):
        if not (path.name in TARGETS or path.name in {x + ".exe" for x in TARGETS} or
                re.search(r"\.(?:a|lib|dll|dylib|so(?:\.[0-9.]+)?)$", path.name)):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(build) or not resolved.is_file():
            continue
        st = resolved.stat()
        if not stat.S_ISREG(st.st_mode):
            continue
        record = {"path": path.relative_to(build).as_posix(), "bytes": st.st_size,
                  "sha256": sha256_file(resolved)}
        if path.is_symlink():
            record["symlink_target"] = resolved.relative_to(build).as_posix()
        result.append(record)
    return result


def execute_plan(plan, args):
    build_root = Path(plan["build_root"])
    # Exclusive creation is also the final no-overwrite check after validation.
    build_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    record = {"schema_version": 1, "plan": plan, "status": "BUILDING",
              "inference_executed": False, "tests_executed": False, "commands": [], "builds": {}}
    record_path = build_root / "build_record.json"
    atomic_json(record_path, record)
    result = 0
    try:
        for step in plan["steps"]:
            log_path = build_root / (step["role"] + "-" + step["phase"] + ".log")
            print(f"{step['role']} {step['phase']}  →  {log_path}", file=sys.stderr, flush=True)
            command = run_cmake_step(step, build_root, args.step_timeout)
            record["commands"].append(command)
            atomic_json(record_path, record)
            if command["returncode"]:
                result = command["returncode"]
                print(f"Failed ({result}). See {build_root / command['log']}", file=sys.stderr)
                record["status"] = "BUILD_FAILED"
                break
        if result == 0:
            # No model or newly built executable is run for this validation.
            after = validate_pair(Path(plan["sources"]["baseline"]["path"]),
                                  Path(plan["sources"]["candidate"]["path"]),
                                  args.manifest.expanduser().resolve(),
                                  args.patches_dir.expanduser().resolve() if args.patches_dir else
                                  args.manifest.expanduser().resolve().parent / "patches",
                                  allow_new_base=args.allow_unvalidated_base)
            require(after == plan["verification"], "Source provenance changed during compilation")
            record["status"] = plan["verification"]["status"]
    except (BuildPairError, OSError) as exc:
        result = 2
        record.update(status="BUILD_FAILED", error=str(exc))
    finally:
        for role in ("baseline", "candidate"):
            build = build_root / role
            record["builds"][role] = {"compiler": compiler_record(build), "artifacts": build_artifacts(build)}
        if result == 0:
            for role in ("baseline", "candidate"):
                found = {Path(x["path"]).name.removesuffix(".exe") for x in record["builds"][role]["artifacts"]}
                if not set(TARGETS) <= found:
                    result = 2
                    record.update(status="BUILD_FAILED", error="Expected compiled tools are missing: " + role)
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["exit_code"] = result
        atomic_json(record_path, record)
    print(json.dumps({"status": record["status"], "record": str(record_path),
                      "inference_executed": False, "exit_code": result}, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--build-root", type=Path, required=True, help="NEW directory outside both checkouts")
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "manifest.json")
    p.add_argument("--patches-dir", type=Path)
    p.add_argument("--jobs", type=int, default=min(4, available_jobs()))
    p.add_argument("--cuda", action="store_true", help="Explicitly enable CUDA compilation (default CPU)")
    p.add_argument("--cuda-arch", help="One architecture, e.g. 120; requires --cuda")
    p.add_argument("--cmake", default="cmake", help="CMake executable name/path")
    p.add_argument("--step-timeout", type=int, default=7200, help="Seconds per configure/build step")
    p.add_argument("--allow-unvalidated-base", action="store_true",
                   help="Reconstruct trusted patches independently on a new explicit base; build only, no semantic PASS")
    run = p.add_mutually_exclusive_group()
    run.add_argument("--execute", action="store_true", help="Actually configure/build; no inference or tests")
    run.add_argument("--dry-run", action="store_true", help="Default: validate and print plan without a build root")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = make_plan(args)
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return execute_plan(plan, args)
    except (BuildPairError, OSError, ValueError) as exc:
        print("build_pair: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
