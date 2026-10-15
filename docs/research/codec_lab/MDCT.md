# Actual MDCT cancellation control

This lane implements the transform mechanism discussed in the codec analogy,
not a chess algorithm. Protocol was committed as `5910801`; executable source
was committed as `36c8f26` before its first execution.

## Operation

An analysis block has 2N real samples, a sine window, and only N MDCT
coefficients. Adjacent analysis blocks start N samples apart. The inverse uses
the matching window and overlap-add. The implementation directly evaluates
the cosine matrix, making the arithmetic easy to inspect rather than fast.
It is not a Vorbis or Opus bitstream implementation.

Each reconstructed block is decomposed into:

    reconstructed_block = window_squared * original_block + alias_residual

We measure the residual of each block and then the sum of residuals in the
overlap. Padding supplies complete boundary context; only the original signal
is scored. The sine-window squared weights sum to one across adjacent blocks.

## Frozen experiment and results

For each N=8 and N=16: 578 signals of length 256, consisting of all 256 positive
and negative impulses, 24 phase/frequency combinations, 10 step edges, and
32 seeded normal signals. The impulse family covers the entire linear input
basis. The tolerance is 1e-10 maximum absolute reconstruction error.

| Variant | N=8 error cases / 578 | N=16 error cases / 578 | Mean relative L2, N=16 |
| --- | ---: | ---: | ---: |
| Signed coefficients, matching windows, overlap-add | 0 | 0 | 4.38e-15 |
| Replace coefficients with absolute magnitudes | 578 | 578 | 1.399 |
| Synthesize with a different window | 578 | 578 | 0.651 |
| Reverse the polarity of alternate blocks | 578 | 578 | 1.290 |
| Round coefficients to steps of 0.25 | 578 | 578 | 0.126 |

The largest isolated alias residual was 2.043 for N=8 and 1.809 for N=16.
After overlap-add, the largest residual was 1.34e-14 and 2.09e-14 respectively.
Maximum complete reconstruction error was 2.05e-14 across both sizes.
The mismatched window retains about 0.380 mean relative L2 error even after
fitting an optimal global output gain, so the failure is not just a scale
convention.

The complete analysis matrix for an individual block has rank N, not 2N.
Nevertheless the overlapping sequence reconstructs the complete signal.
Finite padding adds a boundary block of coefficients; we do not claim a
finite-file storage ratio from the per-block rank.

The positive and negative impulse at sample 8 have exactly equal coefficient
magnitudes and identical magnitude-only decodes, but original signals differ
by L2 distance 2. This is a concrete information-loss collision. It cannot be
repaired downstream without additional information or restrictions on inputs.

## What this establishes

Composition can recover information that an individual transform block cannot
represent alone. The transform, windows, overlap, and relative signs have a
jointly specified reconstruction meaning. Ordinary redundancy or averaging
does not automatically inherit that property. Quantization is a separate
source of error; the cancellation construction does not erase it.

This is a reproduction of known signal-processing mathematics. There is no
chess query, decision algorithm, novel transform, optimized speed result, or
playing-strength result in this lane.

## Reproduce

Requires Python and NumPy. In the current environment:

```sh
/tmp/chen-jpeg2-cpu-venv/bin/python experiments/codec_lab/mdct/run.py \
  --out /tmp/chen-mdct-replay.json
```

Compact [results](../../../experiments/codec_lab/mdct/results.json) bind protocol
and source SHA-256. The main source is [run.py](../../../experiments/codec_lab/mdct/run.py).
Background: [Princen and Bradley, 1986](https://ieeexplore.ieee.org/document/1164954)
and [Vorbis synthesis](https://xiph.org/vorbis/doc/Vorbis_I_spec.html).
