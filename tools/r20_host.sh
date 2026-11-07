#!/usr/bin/env bash
# All generated files and processes belong to the experimental directory.
set -euo pipefail
mode=${1:-first}
kit=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
base=${R20_WORK:-"$HOME/llama-experiment/r20"}
model=${MODEL:-"$HOME/AI_Models/deepseek_4.1_flash/DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00001-of-00007.gguf"}
base=$(realpath -m -- "$base")
run="$base/run"
pin=dde879bc4f954b809892fd0d660515f42536394a
old=${R19_BUILDS:-"$HOME/llama-r19.bXfCIk/run/builds"}
dest="$base/results"
case "$mode" in first|build|diagnose|short|bundle|long32|long128|full128|cancel|maya|results) ;; *)
  echo "Use first, build, diagnose, short, bundle, long32, long128, full128, cancel, maya or results." >&2; exit 2;; esac
case "$base/" in "$kit/"*|"$HOME/llama.cpp/"*|"$HOME/project-maya-test/"*) echo "Choose an isolated R20_WORK" >&2; exit 2;; esac
# Cooperating launches through this wrapper cannot occupy the same GPU together.
mkdir -p -- "$base" "$dest"
exec 9>"$base/.r20-run.lock"
flock -n 9 || { echo "Another R20 wrapper is active in $base" >&2; exit 2; }
build_pair() {
    if [[ -f "$run/builds/build_record.json" ]]; then
        python3 - "$kit" "$run/builds" <<'PY'
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
from r20_bench import provenance
provenance(SimpleNamespace(builds=Path(sys.argv[2]),ngl=18))
PY
        return
    fi
    if [[ -e "$run" ]]; then
        echo "Incomplete run at $run. Logs preserved. Choose a new R20_WORK." >&2; return 2
    fi
    (cd "$kit" && sha256sum --quiet --check SHA256SUMS.txt)
    if [[ ! -d "$base/pinned/.git" ]]; then
        private=$(mktemp -d "$base/source.XXXXXXXX")
        mkdir "$private/pin.git"
        tar -xzf "$kit/vendor/llama-upstream-pin.git.tar.gz" -C "$private/pin.git"
        git clone -q --no-hardlinks "file://$private/pin.git" "$base/pinned"
    fi
    [[ $(git -C "$base/pinned" rev-parse HEAD) == "$pin" ]]
    bash "$kit/tools/p17_build.sh" "$run" "$base/pinned"
}
require_model() {
    [[ -f "$model" ]] || { echo "Missing first GGUF shard: $model" >&2; exit 2; }
    [[ -f "$run/builds/build_record.json" ]] || { echo "Run the build stage first" >&2; exit 2; }
}
historical() {
    if [[ ! -f "$old/build_record.json" ]]; then
        old="$base/historical/run/builds"
        if [[ ! -f "$old/build_record.json" ]]; then
            mkdir -p "$base/historical"
            if [[ ! -d "$base/historical/Llama_Experimental_SOURCE_R19" ]]; then
                unzip -q "$kit/handoff/Llama_Experimental_SOURCE_R19_CUMULATIVE_20261103.zip" -d "$base/historical"
            fi
            bash "$base/historical/Llama_Experimental_SOURCE_R19/tools/p17_build.sh" "$base/historical/run" "$base/pinned"
        fi
    fi
}
stamp=$(date +%Y%m%d-%H%M%S)-$$
common=(--builds "$run/builds" --model "$model" --threads 8 --ngl 18 --moe-cache-mib 12288)
bench() {
    local label=$1
    shift
    python3 "$kit/tools/r20_bench.py" "${common[@]}" --predict 256 --repeats 6 --timeout 86400 --out "$dest/$label-$stamp" "$@"
}
inventory() {
    python3 "$kit/tools/r20_gguf_inventory.py" --source "$base/pinned" --model "$model" --out "$dest/inventory-$stamp.json"
    python3 - "$kit/reports/PUBLISHED_GGUF_HEADERS.json" "$dest/inventory-$stamp.json" <<'PY'
import sys,json
refs,host=(json.load(open(p)) for p in sys.argv[1:])
print("Published/host shard header equality:",refs["shards"]==host["shards"])
print("Logical expert weight bytes per token:",host.get("logical_routed_weight_bytes_per_token"))
PY
}
diag() {
    inventory
    for prof in candidate_all_off r20-full; do
        python3 "$kit/tools/r20_profile.py" "${common[@]}" --ctx 131072 --predict 256 --profile "$prof" --perf --devices --out "$dest/profile-$prof-$stamp"
    done
}
full() {
    historical
    bench bundle --ctx 131072 --bundle r19-compatible --r19-builds "$old" --profile planner_only --profile cpu_iq3_signmask --profile strata_demand_readahead
}
case "$mode" in
  build) build_pair ;;
  first) build_pair; require_model; diag; full ;;
  diagnose) require_model; diag ;;
  short) require_model; bench short --ctx 131072 --profile planner_only --profile cpu_iq3_signmask --profile strata_demand_readahead ;;
  bundle) require_model; full ;;
  long32|long128|full128)
    require_model
    ctx=131072
    case "$mode" in long32) size=(--fill-tokens 32768);; long128) size=(--near-full);; full128) ctx=131584; size=(--fill-tokens 131072);; esac
    bench "$mode" --ctx "$ctx" "${size[@]}" --profile "${R20_PROFILE:-strata_demand_readahead}"
    ;;
  cancel) require_model; python3 "$kit/tools/r20_cancel.py" "${common[@]}" --ctx 131072 --profile "${R20_PROFILE:-strata_demand_readahead}" --out "$dest/cancel-$stamp" ;;
  maya)
    require_model
    : "${MAYA_SRC:?Set MAYA_SRC to the matching Maya v1.1.46 checkout}"
    : "${MAYA_BIN:?Set MAYA_BIN to maya-ds4 built from that revision}"
    historical
    extra=(--r19-builds "$old")
    if [[ -n "${MAYA_DRAFTER:-}" ]]; then
        : "${MAYA_DRAFTER_SHA256:?Set the hash of the companion drafter .bin}"
        extra+=(--drafter "$MAYA_DRAFTER" --drafter-bin-sha256 "$MAYA_DRAFTER_SHA256")
    fi
    if [[ -n "${LLAMA_DSPARK_GGUF:-}" ]]; then
        extra+=(--llama-dspark-gguf "$LLAMA_DSPARK_GGUF")
    fi
    python3 "$kit/tools/r20_maya_ab.py" --model "$model" --builds "$run/builds" --maya-repo "$MAYA_SRC" --maya-bin "$MAYA_BIN" "${extra[@]}" --out "$dest/maya-$stamp" ;;
  results)
    tar --exclude='perf.data*' -czf "$base/R20-results-$stamp.tar.gz" -C "$base" results
    echo "$base/R20-results-$stamp.tar.gz"
    ;;
esac
