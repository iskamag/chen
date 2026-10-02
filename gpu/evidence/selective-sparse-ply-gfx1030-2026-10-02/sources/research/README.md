This holds some of the GC experiments that use 1-bit remembered sets.

To run them, use `make c`, `make cl`, `make hip`, for the C, OpenCL and HIP programs, respectively.

C has all the algorithms, the GPU versions do not have parity.

The single-wave GPU live-set kernel verifies all 256 result bits against a
CPU reference, then measures repeated five-pass latency in one workgroup:

```sh
make hip-scalar
./build/hip/live_set_scalar_hip 128000
# Experimental RDNA wave64 (unsupported by the HIP runtime):
make hip-scalar-wave64
./build/hip/live_set_scalar_hip_wave64 128000
```

Use `128000 --verify-only` to skip timing. Matrices are 256x256; peeling starts
with every object alive. Loads and launch overhead are amortized, so timings
exclude transfers and do not represent fresh-dispatch latency.
