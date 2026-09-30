#!/usr/bin/env bash
set -euo pipefail
gpu_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=${1:-"$gpu_dir/build/libchen_gpu.so"}
mkdir -p -- "$(dirname -- "$output")"
"${HIPCC:-hipcc}" -O3 -std=c++17 --offload-arch=gfx1030 -fPIC -shared \
    -Wall -Wextra -Werror -I"$gpu_dir" -x hip "$gpu_dir/chen_gpu.hip" -o "$output"
