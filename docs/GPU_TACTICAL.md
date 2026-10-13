# Exact GPU tactical observation

The additive [tactical ABI1](../gpu/chen_tactical.h) exports the complete sparse
`chen_tactical_exact_v1` observation from an owned native search arena into
caller-owned Torch storage. It matches the neural worktree's CPU replay schema;
[NativeTacticalProducer](../neural/native_tactical.py) supplies
`TacticalBatch` directly to the recurrent processor. No teacher, future PV,
CPU board download or CPU graph reconstruction occurs in this producer.
Exact mate/draw adjudication stays authoritative in the terminal sidecar.

## Content and ownership

Each row has 32 padded piece slots, 64 mover-oriented squares and 256 padded
legal-action slots. Piece slots follow ascending canonical square; valid actions
follow ascending canonical mover-oriented action ID. Black orientation flips
ranks, not files. Absolute hash, node index, root epoch, history counters and
active slot counts bind observations to their exact source. The history's fourth
counter is **arena search ply**, not source-game ply. Root ingestion must retain
previous hashes rather than resetting a continuing game's repetition history.

Twenty disjoint typed output intervals contain piece/square/action features,
masks, object-square mapping, action IDs, directed sparse edge indices/kinds/
features, material priors and sidecars. See the header for dimensions/dtypes.
The fourteen relations preserve occupancy, empty-square pseudoattacks, own/enemy
attack/support, pins/pinners/checkers and source/destination/victim/action effects.
EP's off-destination victim, castling rook edits and promotion are explicit.
Outgoing pseudoattack masks are computed once per source and reused for counts
and edges. Pin evidence does not erase a pinned piece's pseudoattacks.

The emitter initializes all padding. A conservative finite edge bound is
`2P + 64P + P(P-1) + 2P + P + 7K <= 4992` for P=32 and K=256: occupancy pairs,
pseudoattacks, occupied-target attacks, pin pairs, checkers and three action
input edges plus at most four changed-square edges. Default E=6144; admitted
capacity is 4992..65536. More than 32 pieces or legal-generation overflow rejects
rather than producing a truncated successful observation. This count argument
is not a machine-checked theorem or a performance bound.

Native pointer admission verifies device, alignment, allocation bounds, output
separation and no alias with arena storage. A stale epoch rejects. Preflight
failures preserve payload and result metadata; kernel/runtime failures can leave
partial payload and require discarding the entire failed snapshot. Successfully
returned storage is independent of the arena and survives reset. The Rust engine
and existing symbolic/replay sinks retain their original ABI declarations.

## Executed scope

The [hardware packet](../gpu/evidence/native-tactical-gfx1030-2026-10-03/README.md)
records four actual gfx1030 tests: 45 complete CPU-reference graph rows, all
relations and special moves, complete native children/repetition history,
snapshot lifetime, nine host-admission rejections and one >32-piece rejection.
A width8/steps2 recurrent processor performed resident forward/backward with
finite active gradients. A separate ABI composition test checks bit-identical
old symbolic channels/history over four states. Hot-path dispatch guards reject
Torch host scalar extraction and explicit CPU payload copies; cold oracle checks
copy outputs deliberately. No optimizer updates or strength tests occurred.

The wrapper synchronizes Torch, allocates owned output tensors and waits for the
native private stream/status metadata. Native C allocates no device memory per
export. This is a correctness bridge, **not** captured/asynchronous execution or
allocation-free inference. Serial edge assembly, capacity-wide zeroing and pointer
admission remain timing questions. Unittest body times are not latency benchmarks.
The strict HIP7.2.4 build ran with Torch's ROCm7.0 provider; exact identities are
bound in the packet, without claiming general cross-version compatibility.
