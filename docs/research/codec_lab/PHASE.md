# Alignment-preserving composition: small exact ablation

This experiment tests one operator, not a chess policy: two signed 8×8 maps produce all225 finite-board relative-offset correlations, then the eight knight offsets are summed. No engine, training, move ranking, or tournament is involved.

The protocol was committed as `dc535f1` before execution. Source was frozen as `8c3b654`; an unchanged authoritative rerun after that freeze reproduced deterministic results byte-for-byte. The first execution began before the additional request to freeze source. The committed timing file comes from the post-freeze run. Seed20261015, absolute tolerance1e-9, and all families/variants were fixed in the protocol.

## What is composed

For maps S and T, the requested output is

`C[dy,dx] = sum S[y,x] * T[y+dy,x+dx]`, with only in-board pairs included.

The spatial oracle explicitly slices and multiplies valid overlaps for all offsets−7…7. A separate piece/offset enumeration checks the knight sum.

The transformed operator zero-pads both maps to16×16, forms `conj(FFT(S))*FFT(T)`, then inverse-transforms **that cross-channel product** into the225 requested correlations. It does not reconstruct either input map before computing the interaction. One product computes every relative alignment together. Padding to at least15 per axis prevents different finite-board offsets from aliasing.

The magnitude ablation replaces each channel spectrum by its absolute values before composition. The power ablation retains squared magnitudes and uses their square roots during composition: the same information, with the fairest equivalent phase-erased decoder. The circular control retains complex coefficients but uses8×8 transforms without padding. All variants return the same225 outputs and knight sum.

## Worked collision

Let T contain one target on c2. Let S contain one knight source, either a1 or b1.

- a1→c2 is a knight attack; b1→c2 is not.
- Both source spectra have identical magnitudes mathematically; observed maximum difference was2.22e-16.
- The complete complex cross-spectrum retains the displacement and produces knight counts1 and0.
- The phase-erased operator produces0 for both.

More generally, all4096 source/target impulse pairs share the same two per-channel magnitude/power spectra. Their correct outputs contain225 distinct correlations, and336 pairs are knight attacks. Consequently **no decoder receiving only these two magnitude spectra can recover this query for every pair**. This information-loss statement is stronger than observing one poor decoder. It does not rule out additional cross-channel measurements that explicitly retain relative alignment.

## Deterministic results

Full-output failures compare every one of225 entries, not merely the final knight sum.

| Family | Cases | Spatial | Complex padded16 | Magnitude only | Power only | Complex circular8 |
|---|---:|---:|---:|---:|---:|---:|
| Exhaustive impulse pairs |4096|0|0|4032|4032|4032|
| Sparse binary maps |64|0|0|64|64|64|
| Dense signed maps |64|0|0|64|64|64|
| Edge-only binary maps |64|0|0|64|64|64|

The padded complex variant's maximum error across all4288 cases was2.8422e-14; all knight sums also passed. Detailed per-family scalar errors are in `results.json`.

The boundary example a1→g2 has finite-board knight count0. Circular8 reports1, because it treats a six-file displacement as minus two. Padded16 reports numerical zero (4.50e-18). Keeping phase does not excuse incorrect boundary conditions.

The nonlinear control exhausts128 blocker subsets along b1…h1 for a rook on a1. A fixed unblocked linear ray kernel disagrees with first-blocker-inclusive reach in126 cases. This is deliberately a weak proposed slider operator, not a limitation theorem for spectral representations: blocker-aware nonlinear composition could operate on the same information. Alignment preservation alone does not implement stopping at a blocker.

## Cost and interpretation

Explicit two-channel storage is1024bytes for spatial float64 maps,8192bytes for padded complex128 spectra, or4096bytes for padded magnitude/power arrays. Each full output is1800bytes. These figures exclude FFT scratch space and Python object overhead. This experiment achieves no compression.

`timing.json` records seven repetitions of64 dense signed cases after one warmup, pinned to CPU1. Every timed variant returns all225 entries; spectral timings separately include input transforms or start from cached spectra. These are exploratory Python/NumPy timings. The spatial oracle performs225 Python-level offset iterations while FFT work runs in native NumPy, so these numbers do **not** establish a speed advantage over an optimized spatial or bitboard implementation. There is no hardware or strength extrapolation.

The exact convolution/correlation identity and the translation ambiguity of magnitudes are established mathematics. This experiment verifies finite-board implementation, actual query composition, and concrete chess-geometric counterexamples. It does not establish a novel algorithm, decision sufficiency, useful action selection, approximate phase quantization, or a complete channel interface.

## Codec connection and boundary

JPEG's DCT coefficients are real and signed; JPEG does not store arbitrary complex Fourier phase. The FFT here is a convenient laboratory representation of **relative alignment**, not a proposed literal JPEG encoding of chess. Signed coefficients and correctly coupled transforms matter; per-channel energy alone can erase the relation being queried.

The [Vorbis specification](https://xiph.org/vorbis/doc/Vorbis_I_spec.html) describes an MDCT codec with inverse transform and overlap-add stages. The supplied [Princen/Bradley paper reference](https://ieeexplore.ieee.org/document/1164954) concerns the transform-coding background; its full text was unavailable through the retrieved page. This lane does not implement MDCT, its windows, or time-domain alias cancellation. Perfect reconstruction of a codec and sufficiency for chess decisions are separate questions. The useful transfer examined here is preserving alignment while composing operators, not claiming audio reconstruction guarantees imply chess reasoning guarantees.

## Reproduce

From this worktree:

```sh
taskset -c 1 /tmp/chen-jpeg2-cpu-venv/bin/python experiments/codec_lab/phase/run.py --out /tmp/phase-results.json --timing-out /tmp/phase-timing.json
cmp /tmp/phase-results.json experiments/codec_lab/phase/results.json
```

Requires NumPy; recorded environment Python3.14.8, NumPy2.5.4. Timing is intentionally separate from deterministic results. Files remain small; no large raw traces are emitted.
