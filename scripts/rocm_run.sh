#!/usr/bin/env bash
# Build Rust on the host, HIP in the existing ROCm container; isolate all task files.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo"
container=${CHEN_ROCM_CONTAINER:-cpp-rocm}
mode=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" == help ]]; then
    echo 'Usage: scripts/rocm_run.sh test | bench [ARGS] | uci'
    exit 0
fi
if [[ "$mode" != test && "$mode" != bench && "$mode" != uci ]]; then
    echo 'Expected test, bench, or uci' >&2
    exit 1
fi
cargo build --release --features hip
remote=$(docker exec "$container" mktemp -d /tmp/chen-integrated.XXXXXXXX)
cleanup() { docker exec "$container" rm -rf -- "$remote" >/dev/null; }
trap cleanup EXIT
docker cp gpu "$container:$remote/gpu"
docker cp benches "$container:$remote/benches"
docker cp target/release/chen "$container:$remote/chen"
docker exec "$container" bash "$remote/gpu/build.sh" "$remote/libchen_gpu.so"
if [[ -n "${CHEN_ARTIFACT_DIR:-}" ]]; then
    mkdir -p -- "$CHEN_ARTIFACT_DIR"
    docker cp "$container:$remote/chen" "$CHEN_ARTIFACT_DIR/chen"
    docker cp "$container:$remote/libchen_gpu.so" "$CHEN_ARTIFACT_DIR/libchen_gpu.so"
    docker cp "$container:$remote/gpu" "$CHEN_ARTIFACT_DIR/gpu-source"
    docker exec "$container" hipcc --version > "$CHEN_ARTIFACT_DIR/hipcc.txt"
    docker inspect "$container" --format '{{.Image}}' > "$CHEN_ARTIFACT_DIR/container-image.txt"
    docker exec "$container" python3 "$remote/gpu/probe.py" "$remote/libchen_gpu.so" --max-batch 1 --repetitions 1 > "$CHEN_ARTIFACT_DIR/device-probe.jsonl"
    python3 - "$CHEN_ARTIFACT_DIR" <<'PY_MANIFEST'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = [p for p in root.rglob('*') if p.is_file()]
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root / 'manifest.json').write_text(json.dumps({'target': 'gfx1030', 'files_sha256': manifest}, indent=2) + '\n')
PY_MANIFEST
fi
if [[ "$mode" == test ]]; then
    manifest=$(mktemp /tmp/chen-hip-test.XXXXXXXX)
    cargo test --release --features hip --test hip --no-run --message-format=json > "$manifest"
    executable=$(python3 -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; print(next(x["executable"] for x in rows if x.get("reason")=="compiler-artifact" and x.get("executable") and x.get("target",{}).get("name")=="hip"))' "$manifest")
    rm -- "$manifest"
    docker cp "$executable" "$container:$remote/hip-tests"
    docker exec -e "CHEN_GPU_LIBRARY=$remote/libchen_gpu.so" "$container" "$remote/hip-tests" --ignored --test-threads=1 --nocapture
else
    docker exec -i -w "$remote" -e "CHEN_GPU_LIBRARY=$remote/libchen_gpu.so" "$container" "$remote/chen" "$mode" "$@"
fi
