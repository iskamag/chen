# Actual gfx1030 packed best-open-child cache

`packed_best_open_child_cache_v1` compares sparse baseline `c33fc8f9…`
(committed7243a74) with candidate `a47b8c4e…`. Sources and both executed native
libraries are retained verbatim. The Rust engine and hardware-test binary are
unchanged, hash-bound references to the previous sparse/fusion archives.
No external neural runtime or training call belongs to this experiment.

Both original portable-oracle hardware tests passed per arm. The bounded
follow-up additionally has 33 native cases/1336 per-step comparisons and 48
actual-device mechanism traces. The selector fixture compares admitted parent
indices/offsets, masked reservation counts for every node, immutable node bytes,
and packed-field preservation through1024 increments. Its reference is the
uncached template instantiation; separate public-API comparisons use the exact
preserved sparse baseline library. Configured maximum depth/batch/candidate
bounds are64/256/1024. The deep synthetic fixture reaches actual ancestor
ply63 and leaf ply63 before prospective expansion to64; it is not legal chess.

All72 alternating fixed-work rows agree except timing/rates. Three pairs per
suite give six-search medians137.194866→100.003288ms and
817.526710→405.193452ms. Per-position medians are retained, including the mate
position's small observed1.50%/2.82% slowdowns. No significance claim follows.

Twelve unprofiled one-second processes and six separate ROCm traces provide
5484 independently legal reported PVs. Opening/Kiwipete complete the same
published trajectory in both traced arms; rook traces do unequal work. Host
API waits overlap GPU time. Worker fences remain3N+3 with no search allocations.
ISA/resource observations come from the second concatenated bundle of each
exact timed library, with input hashes unchanged after explicit-output dumping.

## CPU-only replay

Verify the exact `manifest.json` payload inventory/hash/byte maps first.
Extract raw-records.tar.gz into an empty directory and verify every member
against raw-member-index.json. With python-chess available:

```sh
mkdir /tmp/chen-cache-replay
tar -xzf gpu/evidence/selective-cache-gfx1030-2026-10-02/raw-records.tar.gz \
  -C /tmp/chen-cache-replay
PYTHONDONTWRITEBYTECODE=1 results/qa-venv/bin/python \
  gpu/evidence/selective-cache-gfx1030-2026-10-02/sources/host/gpu/analyze_cache_evidence.py \
  /tmp/chen-cache-replay --output /tmp/chen-cache-summary.json
```

The reconstructed summary must be byte-identical to summary.json. Archived
extract_concatenated_fatbin.py independently reconstructs both dumped bundles;
ELF hashes must match isa/extraction-report.json. Replay makes no GPU calls.
The fixture's first compiler failure (host invocation of a device-only helper)
is preserved with source/diagnostics. The corrected fixture checks the mask on
the host and the actual helper on the device. No native failure was replayed.

Private work/arena sizes and replay alias/destructor ownership are unchanged.
Classical count helpers mask low16 bits; external selection retains original
plain-count semantics. Core implementation correspondence is established by
bounded hardware comparisons and source review, not a full machine-code proof.
