#!/usr/bin/env python3
"""Prepare an isolated, auditable patch checkout; never build or alter the source.

All CLI results are JSON.  A successful textual application is not a statement
of API, numerical, performance, or test compatibility.  Forward ports require
an explicit full commit ID and remain UNVALIDATED_NEW_BASE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any


KIT_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_NAME = ".llama-overlay-receipt.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
ADVISORY = (
    "Textual patch application is not API, semantic, numerical, test, or "
    "performance validation. Build and review the isolated candidate separately."
)


class OverlayError(Exception):
    def __init__(self, code: str, message: str, *, incompatible: bool = False,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.incompatible = incompatible
        self.details = details or {}

    def payload(self) -> dict[str, Any]:
        return {
            "status": "INCOMPATIBLE" if self.incompatible else "ERROR",
            "code": self.code,
            "message": self.message,
            **self.details,
        }


def safe_git_env() -> dict[str, str]:
    """Avoid ambient Git routing/configuration and optional index refreshes."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "LC_ALL": "C",
    })
    return env


def git(cwd: Path | None, args: list[str], *, data: bytes | None = None,
        check: bool = True) -> subprocess.CompletedProcess[bytes]:
    command = [
        "git", "-c", f"core.hooksPath={os.devnull}",
        "-c", "core.fsmonitor=false", "-c", "core.autocrlf=false",
        "-c", "core.quotePath=false", "-c", "color.ui=false",
        "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false",
        "-c", "submodule.recurse=false", "-c", "protocol.allow=never",
        "-c", "protocol.file.allow=always", *args,
    ]
    try:
        result = subprocess.run(command, cwd=cwd, env=safe_git_env(), input=data,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
    except OSError as exc:
        raise OverlayError("GIT_UNAVAILABLE", str(exc)) from exc
    if check and result.returncode:
        raise OverlayError("GIT_FAILED", result.stderr.decode("utf-8", "replace").strip(),
                           details={"git_operation": args[0], "returncode": result.returncode})
    return result


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise OverlayError("INVALID_PATH", f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or
            any(p in ("", ".", "..", ".git") for p in path.parts) or
            any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise OverlayError("INVALID_PATH", f"Unsafe {label}: {value!r}")
    if value == RECEIPT_NAME:
        raise OverlayError("INVALID_PATH", "The receipt cannot be a patch target")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OverlayError("INVALID_MANIFEST", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _manifest_from_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeError) as exc:
        raise OverlayError("INVALID_MANIFEST", f"Invalid JSON manifest: {exc}") from exc
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise OverlayError("INVALID_MANIFEST", "schema_version must be integer 1")
    if not isinstance(value.get("release_id"), str) or not value["release_id"].strip():
        raise OverlayError("INVALID_MANIFEST", "release_id must be a nonempty string")
    upstream = value.get("upstream")
    if (not isinstance(upstream, dict) or not isinstance(upstream.get("repository"), str)
            or not upstream["repository"] or not isinstance(upstream.get("base_commit"), str)
            or not COMMIT_RE.fullmatch(upstream["base_commit"])):
        raise OverlayError("INVALID_MANIFEST", "upstream requires repository and a full lowercase base_commit")
    patches = value.get("patches")
    if not isinstance(patches, list) or not patches:
        raise OverlayError("INVALID_MANIFEST", "patches must be a nonempty list")
    ids: set[str] = set()
    names: set[str] = set()
    previous_after: dict[str, str] = {}
    for patch in patches:
        if not isinstance(patch, dict) or not isinstance(patch.get("id"), str) or not patch["id"]:
            raise OverlayError("INVALID_MANIFEST", "Each patch needs an id")
        if patch["id"] in ids:
            raise OverlayError("INVALID_MANIFEST", f"Duplicate patch id: {patch['id']}")
        ids.add(patch["id"])
        filename = _relative_path(patch.get("file"), "patch file")
        if filename in names:
            raise OverlayError("INVALID_MANIFEST", f"Duplicate patch file: {filename}")
        names.add(filename)
        if not isinstance(patch.get("sha256"), str) or not SHA256_RE.fullmatch(patch["sha256"]):
            raise OverlayError("INVALID_MANIFEST", f"Invalid patch sha256: {filename}")
        targets = patch.get("targets")
        if not isinstance(targets, list) or not targets:
            raise OverlayError("INVALID_MANIFEST", f"No targets for {filename}")
        paths: set[str] = set()
        for target in targets:
            if not isinstance(target, dict):
                raise OverlayError("INVALID_MANIFEST", "A target must be an object")
            path = _relative_path(target.get("path"), "target path")
            if path in paths:
                raise OverlayError("INVALID_MANIFEST", f"Duplicate target in patch: {path}")
            paths.add(path)
            for key in ("before_sha256", "after_sha256"):
                if not isinstance(target.get(key), str) or not SHA256_RE.fullmatch(target[key]):
                    raise OverlayError("INVALID_MANIFEST", f"{path}: {key} must be a SHA-256 of an existing regular file")
            if target["before_sha256"] == target["after_sha256"]:
                raise OverlayError("INVALID_MANIFEST", f"No content change declared for {path}")
            if path in previous_after and previous_after[path] != target["before_sha256"]:
                raise OverlayError("INVALID_MANIFEST", f"Broken sequential target hash chain: {path}")
            previous_after[path] = target["after_sha256"]
    extra = value.get("additional_files", [])
    if not isinstance(extra, list) or len(extra) > 20:
        raise OverlayError("INVALID_MANIFEST", "additional_files must be a bounded list")
    destinations: set[str] = set()
    sources: set[str] = set()
    patch_targets = set(previous_after)
    for item in extra:
        if not isinstance(item, dict) or set(item) != {"source", "destination", "sha256"}:
            raise OverlayError("INVALID_MANIFEST", "additional_files needs source/destination/sha256")
        source = _relative_path(item["source"], "extra source")
        destination = _relative_path(item["destination"], "extra destination")
        # Explicit narrow vendor lane; never overwrite arbitrary upstream code.
        vendor_ok = (source in ("modules/engram/row_cache.hpp", "modules/engram/mapped_rows.hpp",
                             "modules/engram/model_cache.hpp", "modules/engram/graph_stage.hpp") and
                     destination == "src/llama-experimental/" + source.rsplit("/", 1)[-1]) or (
                         source in ("modules/moe/upload_pipeline.hpp", "modules/moe/upload_types.hpp", "modules/moe/upload_worker.hpp", "modules/moe/upload_plan.hpp", "modules/moe/upload_scheduler.hpp", "modules/moe/upload_completion.hpp") and
                         destination == "src/llama-experimental/" + source.rsplit("/", 1)[-1]) or (
                         source in ("modules/cuda/hc_post_reuse.cuh", "modules/cuda/mmq_tile_cache.hpp", "modules/cuda/mmq_select_probe.hpp") and
                         destination == "ggml/src/ggml-cuda/llama-experimental/" + source.rsplit("/", 1)[-1]) or (
                         source == "modules/speculative/mtp_verify_window.hpp" and
                         destination == "common/llama-experimental/mtp_verify_window.hpp") or (
                         source == "modules/speculative/mtp_rewind_guard.hpp" and
                         destination == "common/llama-experimental/mtp_rewind_guard.hpp") or (
                         source == "third_party_licenses/LICENSE-project-maya" and
                         destination == "licenses/LICENSE-project-maya") or (
                         source == "modules/cpu/cpu_gate_up.hpp" and
                         destination == "src/llama-experimental/cpu_gate_up.hpp")
        if not vendor_ok:
            raise OverlayError("INVALID_MANIFEST", "extra files are restricted to reviewed vendor paths")
        if source in sources or destination in destinations or destination in patch_targets:
            raise OverlayError("INVALID_MANIFEST", "duplicate/conflicting extra file path")
        if not isinstance(item["sha256"], str) or not SHA256_RE.fullmatch(item["sha256"]):
            raise OverlayError("INVALID_MANIFEST", "extra file needs a SHA-256")
        sources.add(source)
        destinations.add(destination)
    return value


def load_manifest(path: Path | str) -> dict[str, Any]:
    return _manifest_from_bytes(Path(path).read_bytes())


def patchset_sha256(manifest: dict[str, Any]) -> str:
    ordered = [{k: patch[k] for k in ("id", "file", "sha256", "targets")}
               for patch in manifest["patches"]]
    encoded = json.dumps(ordered, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assetset_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest.get("additional_files", []), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_target_hashes(manifest: dict[str, Any], after: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for patch in manifest["patches"]:
        for target in patch["targets"]:
            if after or target["path"] not in result:
                result[target["path"]] = target["after_sha256" if after else "before_sha256"]
    return result


def _regular_file(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise OverlayError("UNSAFE_FILE", f"Symlink not allowed: {current}")
    try:
        mode = current.stat().st_mode
    except FileNotFoundError as exc:
        raise OverlayError("MISSING_FILE", f"File missing: {current}", incompatible=True) from exc
    if not stat.S_ISREG(mode):
        raise OverlayError("UNSAFE_FILE", f"Not a regular file: {current}")
    return current


def _patch_paths(raw: bytes, cwd: Path) -> set[str]:
    # Keep the kit's trust boundary deliberately narrow: existing text files.
    forbidden = rb"(?m)^(?:new file mode |deleted file mode |old mode |new mode |rename from |rename to |copy from |copy to |GIT binary patch|Binary files )"
    if re.search(forbidden, raw):
        raise OverlayError("UNSUPPORTED_PATCH", "Creation, deletion, mode changes, renames, copies and binary patches are not supported")
    result = git(cwd, ["apply", "--numstat", "-z"], data=raw, check=False)
    if result.returncode:
        raise OverlayError("INVALID_PATCH", result.stderr.decode("utf-8", "replace").strip())
    paths: set[str] = set()
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise OverlayError("UNSUPPORTED_PATCH", "Only ordinary text-file changes are supported")
        try:
            path = _relative_path(fields[2].decode("utf-8"), "patch target")
        except UnicodeError as exc:
            raise OverlayError("INVALID_PATCH", "Patch paths must be UTF-8") from exc
        paths.add(path)
    if not paths:
        raise OverlayError("INVALID_PATCH", "Patch has no file changes")
    return paths


def _bundle(manifest_path: Path, patches_dir: Path) -> tuple[dict[str, Any], str, list[bytes]]:
    raw_manifest = manifest_path.read_bytes()
    manifest = _manifest_from_bytes(raw_manifest)
    patches_dir = patches_dir.resolve(strict=True)
    patches: list[bytes] = []
    for patch in manifest["patches"]:
        raw = _regular_file(patches_dir, patch["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != patch["sha256"]:
            raise OverlayError("PATCH_HASH_MISMATCH", f"Patch integrity check failed: {patch['file']}")
        declared = {target["path"] for target in patch["targets"]}
        actual = _patch_paths(raw, patches_dir)
        if actual != declared:
            raise OverlayError("PATCH_TARGET_MISMATCH", f"Patch touches undeclared or missing targets: {patch['file']}",
                               details={"declared": sorted(declared), "actual": sorted(actual)})
        patches.append(raw)
    for extra in manifest.get("additional_files", []):
        entry = _regular_file(manifest_path.parent, extra["source"])
        if entry.stat().st_size > 4*1024*1024 or sha256_file(entry) != extra["sha256"]:
            raise OverlayError("ASSET_HASH_MISMATCH", "Vendor header differs from manifest")
    return manifest, hashlib.sha256(raw_manifest).hexdigest(), patches


def _source_root(source: Path) -> Path:
    if not source.is_dir():
        raise OverlayError("INVALID_SOURCE", "--source must be an existing local Git working tree")
    root = git(source.resolve(), ["rev-parse", "--show-toplevel"]).stdout.decode().strip()
    return Path(root).resolve()


def _reject_executable_filters(source: Path) -> None:
    # Git status/diff can execute a clean/process filter merely to compare a
    # worktree file with its index entry. Hooks/fsmonitor controls do not stop it.
    result = git(source, ["config", "--null", "--get-regexp",
                          r"^filter\..*\.(clean|process|smudge)$"], check=False)
    if result.returncode not in (0, 1):
        raise OverlayError("GIT_CONFIG_FAILED", "Could not inspect effective Git filter configuration")
    commands = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        key, separator, value = record.partition(b"\n")
        if not separator:
            raise OverlayError("GIT_CONFIG_FAILED", "Unexpected Git filter configuration record")
        if value:
            commands.append(key.decode("utf-8", "replace"))
    if commands:
        raise OverlayError("UNSAFE_GIT_FILTER", "Executable Git filters are not supported by the read-only source inspection; use a separate clean upstream clone",
                           details={"filter_keys": commands})


def _source_snapshot(source: Path) -> dict[str, Any]:
    _reject_executable_filters(source)
    # A root config guard cannot see commands in an initialized submodule's
    # local config. Refuse gitlinks before status can inspect them recursively.
    entries = git(source, ["ls-files", "--stage", "-z"]).stdout.split(b"\0")
    if any(entry.startswith(b"160000 ") for entry in entries):
        raise OverlayError("UNSUPPORTED_GITLINK", "P01 does not support source checkouts with Git submodules; no recursive inspection was performed")
    head = git(source, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.decode().strip()
    index_path = Path(git(source, ["rev-parse", "--git-path", "index"]).stdout.decode().strip())
    if not index_path.is_absolute():
        index_path = source / index_path
    tracked = git(source, ["status", "--porcelain=v1", "--untracked-files=no",
                           "--ignore-submodules=none"]).stdout.decode("utf-8", "replace")
    return {"head": head, "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
            "tracked_status": tracked, "tracked_clean": not bool(tracked)}


def _hashes(root: Path, targets: dict[str, str]) -> dict[str, str]:
    return {path: sha256_file(_regular_file(root, path)) for path in sorted(targets)}


def _installed_assets(root: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    return [{"path": x["destination"], "sha256": sha256_file(_regular_file(root, x["destination"]))}
            for x in manifest.get("additional_files", [])]


def _assets_match(root: Path, manifest: dict[str, Any]) -> bool:
    for x in manifest.get("additional_files", []):
        try:
            if sha256_file(_regular_file(root, x["destination"])) != x["sha256"]:
                return False
        except OverlayError:
            return False
    return True


def _install_assets(destination: Path, manifest: dict[str, Any], manifest_path: Path) -> None:
    for x in manifest.get("additional_files", []):
        source = _regular_file(manifest_path.parent, x["source"])
        target = destination / x["destination"]
        # Never replace existing files, directories or symlinks, including
        # an upstream addition that now occupies the private vendor namespace.
        if os.path.lexists(target):
            raise OverlayError("ASSET_DESTINATION_EXISTS", f"Vendor target already exists: {x['destination']}", incompatible=True)
        for part in list(target.parents)[-2::-1]:
            if part == destination.parent:
                continue
            if part.is_relative_to(destination):
                if os.path.islink(part):
                    raise OverlayError("UNSAFE_ASSET_PARENT", f"Symlink directory: {part}")
                if part.exists() and not part.is_dir():
                    raise OverlayError("UNSAFE_ASSET_PARENT", f"Non-directory: {part}")
                part.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        if sha256_file(target) != x["sha256"]:
            raise OverlayError("ASSET_COPY_MISMATCH", f"Vendor asset differs after copying: {x['destination']}")


def _apply_check(root: Path, raw: bytes, patch_id: str) -> None:
    result = git(root, ["apply", "--check", "--whitespace=error-all"], data=raw, check=False)
    if result.returncode:
        raise OverlayError("CONTEXT_CONFLICT", f"Patch {patch_id} cannot be applied exactly; no fuzzy or three-way fallback is permitted",
                           incompatible=True,
                           details={"patch_id": patch_id, "git_error": result.stderr.decode("utf-8", "replace").strip()})


def diff_sha256(root: Path, target_paths: list[str]) -> str:
    raw = git(root, ["diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--", *sorted(target_paths)]).stdout
    return hashlib.sha256(raw).hexdigest()


def inspect(source: Path, manifest_path: Path, patches_dir: Path) -> dict[str, Any]:
    manifest, manifest_hash, patches = _bundle(manifest_path, patches_dir)
    source = _source_root(source)
    snapshot = _source_snapshot(source)
    before = expected_target_hashes(manifest, False)
    after = expected_target_hashes(manifest, True)
    actual = _hashes(source, before)
    state = ("PATCHED" if actual == after and _assets_match(source, manifest)
             else "BASELINE" if actual == before else "DIFFERENT")
    checks = []
    previously_touched: set[str] = set()
    if state != "PATCHED":
        for patch, raw in zip(manifest["patches"], patches):
            targets = {entry["path"] for entry in patch["targets"]}
            if targets & previously_touched:
                checks.append({"id": patch["id"], "status": "DEFERRED_SEQUENTIAL_PATCH",
                               "reason": "An earlier patch changes this file; prepare checks the sequence in the isolated checkout"})
            else:
                result = git(source, ["apply", "--check", "--whitespace=error-all"], data=raw, check=False)
                checks.append({"id": patch["id"], "status": "CHECK_PASSED" if not result.returncode else "INCOMPATIBLE",
                               "git_error": result.stderr.decode("utf-8", "replace").strip() or None})
            previously_touched.update(targets)
    if _source_snapshot(source) != snapshot:
        raise OverlayError("SOURCE_CHANGED", "Source changed during inspection; no reliable inspection result is available")
    return {"operation": "inspect", "status": "INSPECTED", "source": str(source),
            "source_head": snapshot["head"], "tracked_clean": snapshot["tracked_clean"],
            "tree_state": state, "manifest_base": manifest["upstream"]["base_commit"],
            "matches_manifest_base": snapshot["head"] == manifest["upstream"]["base_commit"],
            "requires_explicit_ref": snapshot["head"] != manifest["upstream"]["base_commit"],
            "release_id": manifest["release_id"], "manifest_sha256": manifest_hash,
            "patchset_sha256": patchset_sha256(manifest),
            "assetset_sha256": assetset_sha256(manifest),
            "vendor_files_present": _assets_match(source, manifest),
            "files": [{"path": path, "sha256": digest} for path, digest in actual.items()],
            "patch_checks": checks, "source_preserved": True, "advisory": ADVISORY}


def _receipt(root: Path, manifest: dict[str, Any], manifest_hash: str, commit: str,
             role: str, status: str, application: str, applied_ids: list[str]) -> dict[str, Any]:
    actual = _hashes(root, expected_target_hashes(manifest, False))
    return {"schema_version": 1, "role": role, "status": status, "base_commit": commit,
            "release_id": manifest["release_id"], "manifest_sha256": manifest_hash,
            "patchset_sha256": patchset_sha256(manifest),
            "assetset_sha256": assetset_sha256(manifest),
            "additional_files": _installed_assets(root, manifest) if role != "baseline" else [],
            "files": [{"path": path, "sha256": digest} for path, digest in actual.items()],
            "diff_sha256": diff_sha256(root, list(actual)),
            "application_state": application, "applied_patch_ids": applied_ids,
            "validation_status": "UNVALIDATED", "advisory": ADVISORY}


def _write_receipt(root: Path, receipt: dict[str, Any]) -> None:
    # Exclusive creation: even an unexpected receipt in the upstream tree is not overwritten.
    with (root / RECEIPT_NAME).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def prepare(source: Path, destination: Path, manifest_path: Path, patches_dir: Path,
            ref: str | None = None, baseline_only: bool = False) -> dict[str, Any]:
    manifest, manifest_hash, patches = _bundle(manifest_path, patches_dir)
    source = _source_root(source)
    snapshot = _source_snapshot(source)
    if not snapshot["tracked_clean"]:
        raise OverlayError("SOURCE_DIRTY", "Source tracked files/index must be clean; source was not modified")
    if os.path.lexists(destination):
        raise OverlayError("DESTINATION_EXISTS", "Destination must not exist; nothing will be overwritten")
    if not destination.parent.is_dir():
        raise OverlayError("DESTINATION_PARENT_MISSING", "Destination parent must already exist")
    destination = destination.resolve()
    common = Path(git(source, ["rev-parse", "--git-common-dir"]).stdout.decode().strip())
    common = (source / common).resolve() if not common.is_absolute() else common.resolve()
    if destination.is_relative_to(source) or destination.is_relative_to(common):
        raise OverlayError("UNSAFE_DESTINATION", "Destination must be outside the source working tree and Git directory")
    base = manifest["upstream"]["base_commit"]
    if ref is not None and not COMMIT_RE.fullmatch(ref):
        raise OverlayError("REF_NOT_FULL_SHA", "--ref must be a full lowercase commit SHA, not a branch or tag")
    commit = git(source, ["rev-parse", "--verify", "--end-of-options", f"{ref or 'HEAD'}^{{commit}}"] ).stdout.decode().strip()
    if ref is None and commit != base:
        raise OverlayError("EXPLICIT_REF_REQUIRED", "HEAD differs from the manifest base; review the new base and supply its full SHA explicitly",
                           incompatible=True, details={"source_head": commit, "manifest_base": base})
    role = "baseline" if baseline_only else "candidate"
    applied_ids: list[str] = []
    cloned = False
    try:
        # Local path only. --no-hardlinks and --dissociate avoid sharing the source's object storage.
        git(None, ["clone", "--no-hardlinks", "--no-checkout", "--dissociate",
                   "--no-recurse-submodules", "--template=", "--", str(source), str(destination)])
        cloned = True
        git(destination, ["checkout", "--detach", commit, "--"])
        if os.path.lexists(destination / RECEIPT_NAME):
            raise OverlayError("UNEXPECTED_RECEIPT", "Selected upstream already contains a receipt path; refusing to overwrite it")
        before = expected_target_hashes(manifest, False)
        after = expected_target_hashes(manifest, True)
        actual = _hashes(destination, before)
        if baseline_only:
            if commit == base and actual != before:
                raise OverlayError("BASELINE_HASH_MISMATCH", "Manifest baseline target bytes do not match the selected commit", incompatible=True)
            status = "BASELINE_READY" if commit == base else "UNVALIDATED_NEW_BASE"
            application = "SKIPPED_BASELINE"
        elif actual == after and _assets_match(destination, manifest):
            status = "PATCHED_BASELINE" if commit == base else "UNVALIDATED_NEW_BASE"
            application = "ALREADY_PRESENT"
        else:
            if any(actual[path] == after[path] for path in after):
                raise OverlayError("PARTIALLY_PATCHED", "Only part of the declared final overlay is present; automatic mixed-state repair is prohibited", incompatible=True)
            for patch, raw in zip(manifest["patches"], patches):
                target_before = {t["path"]: t["before_sha256"] for t in patch["targets"]}
                target_after = {t["path"]: t["after_sha256"] for t in patch["targets"]}
                if commit == base and _hashes(destination, target_before) != target_before:
                    raise OverlayError("BEFORE_HASH_MISMATCH", f"Baseline preimage mismatch for {patch['id']}", incompatible=True)
                _apply_check(destination, raw, patch["id"])
                git(destination, ["apply", "--whitespace=error-all"], data=raw)
                applied_ids.append(patch["id"])
                if commit == base and _hashes(destination, target_after) != target_after:
                    raise OverlayError("AFTER_HASH_MISMATCH", f"Patch result disagrees with the trusted manifest: {patch['id']}", incompatible=True)
            if manifest.get("additional_files"):
                _install_assets(destination, manifest, manifest_path)
            changed = set(filter(None, git(destination, ["diff", "--name-only", "-z", "--no-ext-diff", "--no-textconv", "HEAD"]).stdout.decode("utf-8").split("\0")))
            if not changed.issubset(before):
                raise OverlayError("UNDECLARED_CHANGE", "Isolated checkout contains a change outside the manifest targets", incompatible=True)
            if not _assets_match(destination, manifest):
                raise OverlayError("ASSET_INSTALL_FAILED", "Missing or mismatched vendor sources", incompatible=True)
            status = "PATCHED_BASELINE" if commit == base else "UNVALIDATED_NEW_BASE"
            application = "APPLIED"
        receipt = _receipt(destination, manifest, manifest_hash, commit, role, status, application, applied_ids)
        if _source_snapshot(source) != snapshot:
            raise OverlayError("SOURCE_CHANGED", "Source changed during preparation; candidate must not be accepted")
        _write_receipt(destination, receipt)
        return {"operation": "prepare", **receipt, "destination": str(destination),
                "receipt": str(destination / RECEIPT_NAME), "source_preserved": True}
    except OverlayError as exc:
        if cloned and not os.path.lexists(destination / RECEIPT_NAME):
            try:
                failure = _receipt(destination, manifest, manifest_hash, commit, role,
                                   "INCOMPATIBLE", "INCOMPLETE", applied_ids)
                failure["failure"] = exc.payload()
                _write_receipt(destination, failure)
            except (OSError, OverlayError):
                pass  # Preserve the primary error; never repair or delete a failed checkout automatically.
        if cloned:
            exc.details["isolated_destination"] = str(destination)
            exc.details["source_preserved"] = _source_snapshot(source) == snapshot
        raise


class JsonParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise OverlayError("ARGUMENT_ERROR", message)


def main(argv: list[str] | None = None) -> int:
    parser = JsonParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True, parser_class=JsonParser)
    for name in ("inspect", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--manifest", type=Path, default=KIT_ROOT / "manifest.json")
        command.add_argument("--patches-dir", type=Path, default=None,
                             help="Patch filenames are relative to this directory (default: manifest parent / patches)")
        if name == "prepare":
            command.add_argument("--destination", type=Path, required=True)
            command.add_argument("--ref", help="Explicit full commit SHA; required when HEAD is not the manifest base")
            command.add_argument("--baseline-only", action="store_true")
    try:
        args = parser.parse_args(argv)
        patches_dir = args.patches_dir or args.manifest.resolve().parent / "patches"
        if args.operation == "inspect":
            payload = inspect(args.source, args.manifest, patches_dir)
        else:
            payload = prepare(args.source, args.destination, args.manifest, patches_dir,
                              args.ref, args.baseline_only)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except OverlayError as exc:
        print(json.dumps(exc.payload(), ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "code": "IO_OR_DATA_ERROR", "message": str(exc)},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
