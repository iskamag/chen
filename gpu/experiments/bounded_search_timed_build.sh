#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
"${HIPCC:-/opt/rocm/bin/hipcc}" -O3 -std=c++17 --offload-arch=gfx1030 \
  -fPIC -shared -Wall -Wextra -Werror -I gpu -x hip \
  gpu/chen_gpu.hip gpu/experiments/bounded_search_timed.hip -o "${1:?output library required}"
