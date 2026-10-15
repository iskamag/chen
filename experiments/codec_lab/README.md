# Codec composition laboratory

Small CPU experiments about alignment, overlapping operators, and shared
identities. These are not chess bots. The research rationale and source links
are in [the lab note](../../docs/research/CODEC_COMPOSITION_LAB.md).

Each lane has a frozen `protocol.json`, executable experiment, and compact
results. Lane commands and dependencies belong in the lane README/report.
Compare variants only within their declared query contract. No GPU is used.

Start with [the findings](../../docs/research/codec_lab/FINDINGS.md). Python
experiments require NumPy; the overlap experiment requires Rust and the
dependencies pinned by its Cargo.lock. Its runner builds offline, so fetch
dependencies first when using an empty Cargo cache:

```sh
cargo fetch --locked --manifest-path experiments/codec_lab/overlap/Cargo.toml
```

The independent parent audit uses pair enumeration for correlations, the
complete analysis-operator Gram identity for MDCT, a closed intervening-cell
predicate for ray counterexamples, and direct coefficient-domain resource
composition:

```sh
python experiments/codec_lab/audit.py --out /tmp/codec-lab-audit.json
```
