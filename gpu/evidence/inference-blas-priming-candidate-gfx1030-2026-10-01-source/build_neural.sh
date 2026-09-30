#!/usr/bin/env bash
# Reference same-model inference against the installed, version-pinned Torch.
set -euo pipefail
gpu_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=${1:-"$gpu_dir/build/libchen_neural.so"}
torch_dir=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent)')
torch_abi=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
mkdir -p -- "$(dirname -- "$output")"
task_build=$(mktemp -d /tmp/chen-neural-build.XXXXXXXX)
trap 'rm -rf -- "$task_build"' EXIT
"${HIPCC:-hipcc}" -O3 -std=c++17 --offload-arch=gfx1030 -fPIC -c \
    -Wall -Wextra -Werror -I"$gpu_dir" "$gpu_dir/chen_neural_decode.hip" \
    -o "$task_build/decode.o"
"${CXX:-g++}" -O2 -std=c++17 -fPIC -shared -D__HIP_PLATFORM_AMD__ \
    -D_GLIBCXX_USE_CXX11_ABI="$torch_abi" -Wall -Wextra -Werror \
    -I"$gpu_dir" -isystem /opt/rocm/include -isystem "$torch_dir/include" \
    -isystem "$torch_dir/include/torch/csrc/api/include" \
    "$gpu_dir/chen_neural.cpp" "$task_build/decode.o" \
    -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" \
    -Wl,--no-as-needed "$torch_dir/lib/libamdhip64.so" -Wl,--as-needed \
    -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -o "$output"
printf '%s\n' "$torch_dir/lib/libamdhip64.so" > "$output.runtime"
