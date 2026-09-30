#!/usr/bin/env bash
# Copy only Chen GPU sources to an isolated directory in the existing ROCm container.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=${CHEN_ROCM_CONTAINER:-cpp-rocm}
container_dir=$(docker exec "$container" mktemp -d /tmp/chen-gpu.XXXXXXXX)
trap 'docker exec "$container" rm -rf -- "$container_dir"' EXIT
docker cp "$repo_dir/gpu/." "$container:$container_dir/"
docker exec "$container" bash "$container_dir/build.sh"
docker exec "$container" python3 "$container_dir/probe.py" \
    "$container_dir/build/libchen_gpu.so" "$@"
