#!/usr/bin/env python3
"""Freeze isolated full-wave affine candidates; does not initialize a GPU.

Both wave widths use a physical wave per row/output tile. The packed weight
stride changes with width; these libraries cannot replace the production ABI
without a caller that allocates the matching layout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(data):
    return hashlib.sha256(data).hexdigest()


def candidate(source, width, name):
    helpers = ''
    if name.startswith('chen_int8'):
        # RDNA2 ds_bpermute never crosses a 32-lane half. Installed HIP shuffle
        # headers call that builtin directly even when their width argument64.
        source, count = re.subn(r'#pragma unroll\n\s*for\(int step=16;step>0;step/=2\) sum\+=__shfl_down\(sum,step,32\);',
                               'sum=chen_wave_sum(sum);', source)
        assert count == 2, 'integer reduction source changed'
        source, count = re.subn(r'#pragma unroll\n\s*for\(int step=16;step>0;step/=2\) maximum=fmaxf\(maximum,__shfl_xor\(maximum,step,32\)\);',
                               'maximum=chen_wave_max(maximum);', source)
        assert count == 1, 'maximum reduction source changed'
        helpers = '''
__device__ __forceinline__ int chen_wave_sum(int value) {
    #pragma unroll
    for(int step=16;step>0;step/=2) value+=__shfl_down(value,step,32);
    if(CHEN_WAVE==64) value=__builtin_amdgcn_readlane(value,0)+__builtin_amdgcn_readlane(value,32);
    return value;
}
__device__ __forceinline__ float chen_wave_max(float value) {
    #pragma unroll
    for(int step=16;step>0;step/=2) value=fmaxf(value,__shfl_xor(value,step,32));
    if(CHEN_WAVE==64) {
        union Bits { float f; int i; } bits{value},lower{},upper{};
        lower.i=__builtin_amdgcn_readlane(bits.i,0);
        upper.i=__builtin_amdgcn_readlane(bits.i,32);
        value=fmaxf(lower.f,upper.f);
    }
    return value;
}
'''
    # Standalone integer constants only, never uint32_t or block size128.
    replacements = {'32': 'CHEN_WAVE', '31': '(CHEN_WAVE-1)',
                    '16': '(CHEN_WAVE/2)'}
    text = re.sub(r'\b(32|31|16)\b', lambda m: replacements[m[0]], source)
    text = text.replace('namespace {', 'namespace {\n' + helpers, 1)
    kernels = ('linear<true>', 'dynamic_linear<false>', 'scalar<true>',
               'dynamic_linear<true>') if name.startswith('chen_int8') else ('affine<false>', 'affine<true>')
    cases = '\n'.join(f'case {i}: function = reinterpret_cast<const void*>({k}); break;'
                      for i, k in enumerate(kernels))
    admission = '''
// Experimental admission only; initializes the HIP runtime when called.
extern "C" int32_t chen_wave_resources(uint64_t kind,uint64_t block,uint64_t* out) {
    if(!out || !block || block>1024) return int32_t(hipErrorInvalidValue);
    const void* function=nullptr;
    switch(kind) { CASES default: return int32_t(hipErrorInvalidValue); }
    hipFuncAttributes attr{};
    hipError_t rc=hipFuncGetAttributes(&attr,function);
    if(rc!=hipSuccess) return int32_t(rc);
    int active=0;
    rc=hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,int(block),0);
    if(rc!=hipSuccess) return int32_t(rc);
    out[0]=attr.numRegs; out[1]=attr.sharedSizeBytes; out[2]=attr.localSizeBytes;
    out[3]=attr.maxThreadsPerBlock; out[4]=active; out[5]=CHEN_WAVE;
    return int32_t(hipSuccess);
}
'''.replace('CASES', cases)
    return (f'// Isolated experimental full-wave mapping; physical wave {width}.\n'
            f'#define CHEN_WAVE {width}\n' + text + admission)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=1, mapping='one physical wave per row/output tile',
                    padding='round_up_N_to_physical_wave_width',
                    source_inputs={}, generated={}, build_commands=[])
    for name in ('chen_int8_linear.hip', 'float_linear_reference.hip'):
        data = (root / name).read_bytes()
        manifest['source_inputs'][name] = sha(data)
        (output / ('baseline-' + name)).write_bytes(data)
        for width in (32, 64):
            filename = f'wave{width}-{name}'
            generated = candidate(data.decode(), width, name).encode()
            (output / filename).write_bytes(generated)
            manifest['generated'][filename] = sha(generated)
            library = f'libwave{width}-{name.removesuffix(".hip")}.so'
            flag = '-mwavefrontsize64' if width == 64 else '-mno-wavefrontsize64'
            manifest['build_commands'].append([
                '/opt/rocm/bin/hipcc', '-O3', '-shared', '-fPIC', '--offload-arch=gfx1030',
                flag, filename, '-o', library])
    manifest['stage_script_sha256'] = sha(Path(__file__).read_bytes())
    (output / 'source-binding.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
