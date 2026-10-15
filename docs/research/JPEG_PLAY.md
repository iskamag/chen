# Historical JPEG v1 experiments

The canonical `experiments/jpeg_play` entry point now runs the direct learned
channel engine described in [JPEG2.md](JPEG2.md). It requires an explicit model:

```sh
cargo build --release --manifest-path experiments/jpeg_play/Cargo.toml
experiments/jpeg_play/target/release/chen-jpeg-play uci --model /path/to/policy.json
```

The minimax and PVS-wrapper implementations below are archived under
`experiments/jpeg_legacy_v1/`. They are historical controls and do not implement
the intended decision architecture. Previously frozen executables and results
retain their original identities. The new runtime has no search fallback;
its strength must be measured independently.

This adds two explicitly different UCI policies, without changing Chen's
production defaults. Neither establishes a 2500 Elo engine or an architectural
breakthrough. Match results, rather than the name, must determine which is
strongest among the tested candidates.

Build and run:

```sh
cargo build --release --manifest-path experiments/jpeg_legacy_v1/Cargo.toml
experiments/jpeg_legacy_v1/target/release/chen-jpeg-legacy-v1 --mode pure
experiments/jpeg_legacy_v1/target/release/chen-jpeg-legacy-v1 --mode hybrid
```

Both expose `Policy` (`pure`/`hybrid`), `Hash` (default 16 MiB), and one fixed
CPU thread. Opening-book use is disabled. `go movetime 1000` leaves a 35 ms
internal margin, with the relational prepass limited to the earlier of 25 ms
and the remaining deadline. Normal clock/increment controls are also supported.
The UCI process handles `stop` concurrently, preserves previous-position hashes
from `position ... moves ...`, and retains the conventional searcher's TT between
hybrid searches. The pure policy does not use that conventional searcher. Frozen v1 telemetry
reports `prepass_candidates`, `known_safe`, and `known_mate_losses`; the number
unresolved by the opposing-mate gate is exactly
`prepass_candidates - known_safe - known_mate_losses`. That count is derivable
from every recorded decision without changing the frozen executable.

## Shared exact prepass

The counterfactual codec supplies every legal root move, sparse edits, and exact
post-action attack answers. `Preferences` supplies the existing untuned
piece-square/material/exposure ranking. The shared reply relation then certifies:

- Root move checkmate and stalemate, including the original missed mate-in-one.
- Whether each root move permits an immediate opposing checkmate. Only checking
  opponent actions need complete reply exhaustion for this obligation.
- Whether a child meets Chen's draw-clock, insufficient-material or repetition
  conditions, with mate taking priority over a draw-clock transition.

"Known safe" in telemetry means *certified not to allow opponent mate-in-one*,
not general tactical safety. A proven mating root move is selected immediately.
Otherwise a certified mate-in-one loss is excluded whenever a certified safe
move exists. If the prepass times out, unresolved alternatives remain unknown;
they never become negative certificates. A legal static fallback is retained
before proof work starts. When at least one certified safe move exists,
unresolved alternatives are also excluded from the final root selection.

Each shared classification call has a 4,096 operator-word budget. Budget
exhaustion returns `Unknown`. Deadlines are checked between relation calls and
minimax nodes, so a single bounded relation call is not preempted midway through
its own instruction sequence. The 35 ms margin is deliberate protection against
that granularity and conventional search polling overhead; wall-clock compliance
is still checked separately by the match harness.

## `pure`: JPEG relation minimax

This is a playable relation-based policy with iterative minimax that is
**full-width only after the conservative root safety gate**. It is not a static
policy, beam search, alpha-beta implementation, or delegated Chen search. Every
completed iteration visits all *eligible* root alternatives and all legal
opponent continuations. There is no numerical width cap. However, when at least
one certified safe root exists, the gate excludes **both proven immediate-mate
losses and unresolved root candidates**. An incomplete prepass can therefore
omit a sole quiet defense without proving that candidate bad. The remaining
minimax values are conditional on that eligible root set; this is not full-width
minimax over all legal root moves. Finding a proven immediate mate can also end
a node early.

Each node obtains its legal alternative relation from `CodecPosition`. Shared
reply classification distinguishes terminal consequences. A depth-one frontier
reduces all legal move preferences, overriding heuristic values with proven
checkmate/stalemate and draw consequences. Greater depths perform full minimax
over these relations. The untuned geometric exposure score is a frontier
heuristic, not an exchange proof. There is no capture quiescence or TT in this
pure implementation. An interrupted iteration is discarded; the last fully
completed root iteration remains the choice.

This still uses conventional game-tree minimax reasoning. The codec and shared
quantifiers are the position/consequence mechanisms; they have not removed the
need for continuation reasoning. Scalar `Position` copies/makes are used when
moving between relation roots. It would be inaccurate to call this a searchless
engine or a complete implementation of the broader JPEG proposal.

The separate microblock/pyramid and persistent causal experiments are **not
integrated into this playable policy**. This bot tests the counterfactual codec
and shared reply proofs. It reconstructs a local relation at successive game-tree
nodes; it does not maintain a shared representation of equivalent continuations
across those nodes. Learned/lossy strategic decision channels are also absent.
Its reported depth is nominal minimax half-moves, not full chess moves, and
terminal reply proofs can inspect consequences beyond that nominal frontier.
Depth four in the one-second starting-position smoke is not a guarantee of
depth four in complex positions or at a half-second budget.

## `hybrid`: JPEG-assisted Chen PVS

The direct [half-second stock-Chen comparison](JPEG_VS_STOCK_20261015.md) found
no demonstrated improvement: 5 wins, 4 draws and 7 losses; no overrides of 813
conventional search decisions; identical choices on 32 same-position probes.
Treat this policy as an experimental safety wrapper, not a validated upgrade.

The same exact relational prepass is followed by the existing conventional
`Searcher` using Tapered evaluation, PVS, TT, quiescence, move ordering, and a
25 cp aspiration window. It receives only the time left before the common
internal deadline, so the prepass does not add a second time allowance.

This is deliberately named **JPEG-assisted Chen PVS**. Its ordinary game-tree
search, evaluation, and TT are existing Chen mechanisms. A conventional result
is accepted only if it respects a completed immediate-mate safety certificate;
otherwise the prepass's certified-safe fallback is used. This is an honest
hybrid strength control, not evidence that the radical representation itself
has achieved conventional engine strength.

## Verification and frozen executable

The frozen v1 executable used for subsequent variant selection is SHA-256:

`d3820f466ed095ee81a4a7cbc23dad2f40850886936c879c8ffc2d913637ef09`

Source SHA-256:

`5b93330aa1a8865ec0b6c1de32ce910fce0770e509c0e99756ea83e64db4abe9`

Commands:

```sh
cargo test --release --manifest-path experiments/jpeg_legacy_v1/Cargo.toml
cargo clippy --release --manifest-path experiments/jpeg_legacy_v1/Cargo.toml -- -D warnings
```

All four focused release tests pass. They cover mate-in-one for both colors even
at draw clock 100; explicit detection/rejection of `g2g4` allowing Fool's Mate
and its color mirror; preserved repetition history and 99-to-100 draw-clock
transitions; malformed position input; and an exhausted deadline returning an
unknown certificate with a legal fallback rather than a mate claim.

Independent python-chess UCI smoke before the final name/cleanup freeze found:

| Policy | Start-position completed depth | Move wall time | White/black mate-in-one |
| --- | ---: | ---: | --- |
| Pure relation minimax | 4 | 0.967 s | Both immediate mates, under 1 ms |
| JPEG-assisted Chen PVS | 7 | 0.967 s | Both immediate mates, under 1 ms |

Those are local smoke observations, not strength measurements. Root/bands agents
own the independent final-binary special-rule smoke, paired pure-versus-hybrid
selection, Stockfish matches, complete move audits, and video artifacts. The
initial hybrid v0 was frozen separately for its first Stockfish pair; v0 results
must not be silently attributed to v1. Both executable identities should remain
in the match manifests. No GPU code is required or executed.
