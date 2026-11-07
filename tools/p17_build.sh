#!/usr/bin/env bash
# Build two isolated CUDA checkouts at the reviewed commit. Ubuntu / RTX 5090.
set -euo pipefail
mode=--execute
if [[ "${1:-}" == --prepare-only ]]; then
    mode=--dry-run
    shift
fi
if (( $# < 1 || $# > 2 )); then
    echo "Usage: bash tools/p17_build.sh [--prepare-only] NEW_RUN_DIRECTORY [LOCAL_LLAMA_REPO]" >&2
    exit 2
fi
kit=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
run=$(realpath -m -- "$1")
if ! command -v nvcc >/dev/null && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
for cmd in git python3 cmake c++ nvcc sha256sum; do
    command -v "$cmd" >/dev/null || { echo "Missing tool on PATH: $cmd" >&2; exit 2; }
done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ required")'
if ! nvcc --list-gpu-code | grep -x 'sm_120' >/dev/null; then
    echo "This nvcc cannot compile sm_120 for the RTX 5090. Check PATH and CUDA Toolkit." >&2
    exit 2
fi
pin=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream"]["base_commit"])' "$kit/manifest.json")
url=https://github.com/ggml-org/llama.cpp.git
if (( $# == 2 )); then
    repo=$(git -C "$2" rev-parse --show-toplevel)
    git -C "$repo" cat-file -e "$pin^{commit}" || { echo "Pinned commit missing in $repo" >&2; exit 2; }
    url=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().as_uri())' "$repo")
    case "$run/" in "$repo/"*) echo "Choose a run directory outside the source checkout" >&2; exit 2;; esac
fi
case "$run/" in "$kit/"*) echo "Choose a run directory outside the kit" >&2; exit 2;; esac
(cd "$kit" && sha256sum --check --quiet SHA256SUMS.txt)
mkdir -- "$run"
echo "Run directory: $run"
{
    python3 --version
    git --version
    cmake --version
    nvcc --version
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv || true
    fi
} > "$run/preflight.txt"
git init -q "$run/upstream"
git -C "$run/upstream" fetch --no-tags --depth=1 "$url" "$pin"
git -C "$run/upstream" checkout -q --detach FETCH_HEAD
test "$(git -C "$run/upstream" rev-parse HEAD)" = "$pin"
for role in baseline candidate; do
    extra=()
    if [[ "$role" == baseline ]]; then extra+=(--baseline-only); fi
    if ! python3 "$kit/tools/overlay.py" prepare --source "$run/upstream" --ref "$pin" \
        --destination "$run/$role" "${extra[@]}" > "$run/$role-prepare.json"; then
        cat "$run/$role-prepare.json" >&2
        exit 1
    fi
done
python3 "$kit/tools/build_pair.py" --baseline "$run/baseline" --candidate "$run/candidate" \
    --build-root "$run/builds" --cuda --cuda-arch 120 "$mode"
