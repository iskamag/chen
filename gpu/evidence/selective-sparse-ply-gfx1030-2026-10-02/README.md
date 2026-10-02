# Actual classical sparse-ply gfx1030 experiment

Variant: `sparse_ply_single_thread_admitted_ancestors_v1`.
The RX 6800/ROCm 7.2.4 experiment compares preserved fused library
`04447c455f33fc7eca1e119664584dc70fa3a91cba05b3ba3cb35908e60451d2`
with candidate
`c33fc8f97e3fd546af72a3e3f22a45ffb57082a6071aa7b07066ada6d50c14cb`.
There are no neural runtime or training calls in this experiment.

`summary.json` is reconstructed from the raw fixed-work rows, native snapshots,
mechanism fixture output and independent ROCm CSV traces. `artifacts.json`
binds executed binaries and the earlier frozen Rust source closure. Each
candidate compilation input is preserved verbatim in `sources/candidate`;
the baseline's exact source closure is in `sources/baseline`. The active public
header subsequently received a comment-only ordering clarification; its frozen
compiled copy remains unchanged here. `classical-sparse.patch` records the
actual production change against the committed fused implementation.

The equal-work tests comprise both original portable-oracle hardware tests per
arm, 33 native cases / 1,336 per-step metadata+score+PV comparisons, and seven
maximum mechanism fixtures through ancestor depth64/batch256 with guard words.
All 72 alternating fixed-work rows agree except timing/rates. Six-search
medians are 143.180938→135.234465ms for the original bounded workload, and
1593.305559→818.045340ms for the 50,000-node workload. Three pairs per workload
and a small authored suite do not establish significance or playing strength.

Twelve separate one-second UCI observations and six isolated trace runs produce
3,916 independently legal reported PVs. The trace arms perform unequal work;
node counts, kernel/API duration, fence counts and no-search-allocation evidence
are retained without treating host waits as additive device time. Selection
now dominates the sparse classical trace. GPU work was exclusive and all
follow-up processes ended with exit0. CPU5 affinity avoids the earlier root
match core; shared system power/cache were uncontrolled.

## Replay without GPU

Verify every `manifest.json` payload hash/size before extraction. Extract the
raw archive into an empty directory; its `raw-member-index.json` binds every
member individually. With python-chess available:

```sh
mkdir /tmp/chen-sparse-replay
tar -xzf gpu/evidence/selective-sparse-ply-gfx1030-2026-10-02/raw-records.tar.gz \
  -C /tmp/chen-sparse-replay
PYTHONDONTWRITEBYTECODE=1 results/qa-venv/bin/python \
  gpu/analyze_selective_sparse.py /tmp/chen-sparse-replay \
  --output /tmp/chen-sparse-replayed-summary.json
```

The reconstructed summary must be byte-identical to archived `summary.json`.
The archived analyzer and helper sources can also be copied together preserving
`sources/host/gpu` and `sources/host/scripts` if the active sources change.
This replay checks retained evidence; it does not rerun the native GPU tests.

## Code-object provenance and preserved corrections

`raw-records.tar.gz` contains `isa/commands.json`, hash-before/hash-after
records, both concatenated HIP bundles, extracted ELF files, full disassembly,
metadata notes and the completion-kernel subsets. `llvm-objcopy` was given an
explicit separate output `.so` so dumping the section did not rewrite the timed
input. The first bundled translation unit is not the chess rules; the rules
bundle starts at byte12288 in both dumped fatbins. The second ELF contains
`selective_complete_classical`. `sources/host/extract_concatenated_fatbin.py`
reconstructs every ELF from bundle descriptors without modifying its input;
its extracted hashes must match `isa/extraction-report.json`.

Completion metadata is VGPR23 in both arms, SGPR33→42, wave32, scratch0/LDS0.
Static completion opcodes total370→553; this is not dynamic instruction count
or occupancy. Local bitmatrices sources read during tuning are retained in
`sources/research`; runtime research bindings identify the exact official RDNA2
XML and architecture PDF read. The separate RDNA study preserves the full ISA.

The archive retains the fixture's first compiler failure and exact source
(field-order/narrowing in the fixture constructor, no production change).
It also retains the first native history-case attempt with halfmove zero,
which could not reach historical draw adjudication. The corrected driver sets
halfmove two and requires a closed zero-score terminal root. The first driver
source is hash-verified against the earlier UCI/trace declarations, and the
corrected source is retained separately. No native failure was replayed.

Actual hardware validation remains bounded: no rating, equal-time strength,
external neural execution or long-run fault-recovery conclusion follows.
