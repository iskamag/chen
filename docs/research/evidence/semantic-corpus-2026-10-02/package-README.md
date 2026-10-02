# Exact semantic corpus

This package keeps simulator state, full repetition history and complete legal
action groups separate from neural input features. It is an experimental data
interface; it does not select or train a model.

`packing.pack_audited_split` publishes one split from the frozen packet-v2 teacher
archive. Callers must provide an independently pinned serialization seal with
source/archive hashes and allocation limits. Publication checks every label join,
PV, score interval and history, cold-validates the output, then atomically renames
without replacing an existing corpus. Its four files are `manifest.json`,
`payload.bin`, `provenance.json` and `schedule.json`.

`packing.read_pack` verifies bounded metadata, aligned typed tables and all source
witnesses without importing Torch. `purpose='train'` rejects heldout splits.
`scripts/audit_semantic_corpus.py` independently reconstructs the wire format and
uses `examples/corpus_oracle.rs` to check the actual Rust rule states, histories,
terminal decisions and every root's full legal expansion.

`resident.load_resident` first runs CPU admission, then creates an owned tensor
blob and appends the fixed structural schedule indices before its single packed
`Tensor.to` call. Public batches own their gathered storage. Root batches contain
whole action sets grouped by exact legal count; there is no padding or sampled
replacement of siblings. Histories remain sidecars outside schema-1 features.

ROCm use requires a separately pinned admission record and actual `gfx1030`
hardware. Validation requires a frozen checkpoint gate; test additionally requires
a frozen selection gate. These gates concern access to model evaluation data;
mechanical serialization/audit does not calculate model-quality metrics. The
structural schedule is not a sealed learning configuration.

Aligned corpus views are not legal native replay-export sinks: the existing
native bridge requires independent complete base allocations. CPU corpus ingress
and native GPU generation have separate provenance and transfer boundaries.

Run synthetic CPU checks using an environment with python-chess and Torch:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  experiments.semantic_corpus.tests.test_packing \
  experiments.semantic_corpus.test_resident \
  scripts.tests.test_audit_semantic_corpus
```

The detailed state/frame/ownership contract and evidence are linked from
[the research checkpoint](../../docs/research/STATUS.md).
