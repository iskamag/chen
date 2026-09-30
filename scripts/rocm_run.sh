#!/usr/bin/env bash
# Build Rust on the host, HIP in the existing ROCm container; isolate all task files.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo"
container=${CHEN_ROCM_CONTAINER:-cpp-rocm}
mode=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" == help ]]; then
    echo 'Usage: scripts/rocm_run.sh test | neural-test | bench [ARGS] | compare [ARGS] | gpu-perft [ARGS] | uci'
    exit 0
fi
if [[ "$mode" != test && "$mode" != neural-test && "$mode" != bench && "$mode" != uci && "$mode" != gpu-perft && "$mode" != compare ]]; then
    echo 'Expected test, neural-test, bench, compare, gpu-perft, or uci' >&2
    exit 1
fi
cargo build --release --features hip
remote=$(docker exec "$container" mktemp -d /tmp/chen-integrated.XXXXXXXX)
cleanup() { docker exec "$container" rm -rf -- "$remote" >/dev/null; }
trap cleanup EXIT
docker cp gpu "$container:$remote/gpu"
docker cp benches "$container:$remote/benches"
docker cp target/release/chen "$container:$remote/chen"
if [[ "$mode" == compare ]]; then
    docker cp scripts/resident_bench.py "$container:$remote/resident_bench.py"
fi
if [[ "$mode" == neural-test ]]; then
    neural_worktree=${CHEN_NEURAL_WORKTREE:-"$repo/../chen-neural"}
    qa_python=${CHEN_QA_PYTHON:-python3}
    chess_source=$("$qa_python" -c 'import chess; from pathlib import Path; print(Path(chess.__file__).resolve().parent)')
    docker cp "$neural_worktree/neural" "$container:$remote/neural"
    docker cp "$chess_source" "$container:$remote/chess"
fi
docker exec "$container" bash "$remote/gpu/build.sh" "$remote/libchen_gpu.so"
neural_env=()
if [[ -n "${CHEN_NEURAL_MODEL:-}" ]]; then
    if [[ ! -f "$CHEN_NEURAL_MODEL" || ! -f "$CHEN_NEURAL_MODEL.json" ]]; then
        echo 'CHEN_NEURAL_MODEL requires an existing archive and .json sidecar' >&2
        exit 1
    fi
    docker cp "$CHEN_NEURAL_MODEL" "$container:$remote/model.pt"
    docker cp "$CHEN_NEURAL_MODEL.json" "$container:$remote/model.pt.json"
    docker exec "$container" bash "$remote/gpu/build_neural.sh" "$remote/libchen_neural.so"
    torch_runtime=$(docker exec "$container" cat "$remote/libchen_neural.so.runtime")
    # Prebind the producing runtime before any UCI backend can create HIP state.
    neural_env=(-e "CHEN_NEURAL_MODEL=$remote/model.pt" -e "CHEN_NEURAL_LIBRARY=$remote/libchen_neural.so" \
        -e "CHEN_NEURAL_HIP_RUNTIME=$torch_runtime" -e "LD_PRELOAD=$torch_runtime" \
        -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1)
    for execution_key in CHEN_NEURAL_EXECUTION CHEN_NEURAL_GRAPH_SHAPES CHEN_NEURAL_GRAPH_BYTES; do
        if [[ -v "$execution_key" ]]; then
            neural_env+=(-e "$execution_key=${!execution_key}")
        fi
    done
fi
if [[ -n "${CHEN_ARTIFACT_DIR:-}" ]]; then
    mkdir -p -- "$CHEN_ARTIFACT_DIR"
    docker cp "$container:$remote/chen" "$CHEN_ARTIFACT_DIR/chen"
    docker cp "$container:$remote/libchen_gpu.so" "$CHEN_ARTIFACT_DIR/libchen_gpu.so"
    docker cp "$container:$remote/gpu" "$CHEN_ARTIFACT_DIR/gpu-source"
    if [[ -n "${CHEN_NEURAL_MODEL:-}" ]]; then
        for neural_artifact in model.pt model.pt.json libchen_neural.so libchen_neural.so.runtime; do
            docker cp "$container:$remote/$neural_artifact" "$CHEN_ARTIFACT_DIR/$neural_artifact"
        done
        docker exec -i "${neural_env[@]}" "$container" python3 - "$torch_runtime" > "$CHEN_ARTIFACT_DIR/neural-execution.json" <<'PY_EXECUTION'
import hashlib, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
keys = ('CHEN_NEURAL_EXECUTION', 'CHEN_NEURAL_GRAPH_SHAPES', 'CHEN_NEURAL_GRAPH_BYTES',
        'CHEN_NEURAL_HIP_RUNTIME', 'LD_PRELOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
print(json.dumps({'requested_environment': {key: os.environ[key] for key in keys if key in os.environ},
                  'producing_hip_runtime': str(path),
                  'producing_hip_runtime_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'scope': 'Requested policy; bench neural_execution records admitted mode and limits.'}, indent=2))
PY_EXECUTION
    fi
    if [[ "$mode" == compare ]]; then
        docker cp "$container:$remote/resident_bench.py" "$CHEN_ARTIFACT_DIR/resident_bench.py"
    fi
    if [[ "$mode" == neural-test ]]; then
        docker cp "$container:$remote/neural" "$CHEN_ARTIFACT_DIR/neural-source"
        docker cp "$container:$remote/chess" "$CHEN_ARTIFACT_DIR/chess-oracle"
    fi
    docker exec "$container" hipcc --version > "$CHEN_ARTIFACT_DIR/hipcc.txt"
    docker inspect "$container" --format '{{.Image}}' > "$CHEN_ARTIFACT_DIR/container-image.txt"
    docker exec "${neural_env[@]}" "$container" python3 "$remote/gpu/probe.py" "$remote/libchen_gpu.so" --max-batch 1 --repetitions 1 > "$CHEN_ARTIFACT_DIR/device-probe.jsonl"
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
    for test_target in hip device; do
        manifest=$(mktemp /tmp/chen-hip-test.XXXXXXXX)
        cargo test --release --features hip --test "$test_target" --no-run --message-format=json > "$manifest"
        executable=$(python3 -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; print(next(x["executable"] for x in rows if x.get("reason")=="compiler-artifact" and x.get("executable") and x.get("target",{}).get("name")==sys.argv[2]))' "$manifest" "$test_target")
        rm -- "$manifest"
        docker cp "$executable" "$container:$remote/$test_target-tests"
        docker exec -e "CHEN_GPU_LIBRARY=$remote/libchen_gpu.so" "$container" "$remote/$test_target-tests" --ignored --test-threads=1 --nocapture
    done
elif [[ "$mode" == neural-test ]]; then
    docker exec -w "$remote" -e "CHEN_GPU_LIBRARY=$remote/libchen_gpu.so" -e CHEN_TEST_DEVICE=cuda -e CHEN_DYNAMICS_GPU=1 -e CHEN_BRIDGE_TEST_GPU=1 -e CHEN_TEACHER_TEST_GPU=1 -e CHEN_VALUE_STUDY_TEST_GPU=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 "$container" python3 -m unittest discover -s neural/tests -v
elif [[ "$mode" == compare ]]; then
    docker exec -w "$remote" -e "CHEN_GPU_LIBRARY=$remote/libchen_gpu.so" -e "CHEN_SOURCE_COMMIT=$(git rev-parse HEAD)" -e "CHEN_SOURCE_STATUS=$(git status --short)" "$container" python3 resident_bench.py "$@" --output resident-comparison.json
    result_dir=${CHEN_RESULTS_DIR:-results/resident-comparison}
    mkdir -p -- "$result_dir"
    docker cp "$container:$remote/resident-comparison.json" "$result_dir/resident-comparison.json"
else
    docker exec -i -w "$remote" "${neural_env[@]}" -e "CHEN_GPU_LIBRARY=$remote/libchen_gpu.so" "$container" "$remote/chen" "$mode" "$@"
fi
