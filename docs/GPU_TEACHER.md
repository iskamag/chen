# Resident classical search teacher

`gpu/chen_teacher.h` defines a separate teacher ABI1. It exports completed
parent minimax labels directly into caller-owned Torch GPU buffers for the same
existing child-edge selection used by replay export. Sibling rows therefore
share labels for one source parent. The three outputs are exact int32
side-to-move scores, int64 canonical chosen legal actions, and int32 parent ply.
There is no material-proxy label or game-outcome claim.

The 48-byte sink holds version, descriptor size, device, zero reserved field,
capacity, and three GPU addresses. The 32-byte result holds row count, epoch,
HIP event span, evaluator selector, and label kind1: scored resident finite-tree
classical minimax. Evaluator -1 means no selector was locked (terminal-only
score work); 0 is Material and 1 Tapered. External learned evaluators are rejected
by this classical-label API.

A source parent must be expanded, nonterminal, and scored. A read-only GPU
preflight validates the selected edge's parent/child identity, every immediate
child score, the first maximum of negated child scores, the stored best-child
ID, and equality of that maximum with the stored parent score. This rejects
incomplete or unreduced parents before writing any caller output. It preserves
board, node, legal move, and legal count storage. Scratch/timing telemetry may
change. Source parents may have terminal descendants, whose exact mate/draw
scores participate in the backup.

Mate terminals score `-30000 + absolute_root_ply`; backups negate child scores.
Static scores are CP clamped to +/-29000. The exported parent ply makes the mate
origin explicit: a positive mate score from a fresh CPU search at that parent
loses `parent_ply`, and a negative mate score gains it. Canonical action encoding
uses the parent's mover, including black rank XOR56. Ties preserve the first
move in the Rust legal-generation order, rather than sorted canonical support.

The C API labels the actual resident tree. Arbitrarily constructed trees may
have different static-leaf horizons. It makes no uniform-depth claim. A producer
that requires a uniform depth must record and validate its full-width BFS
expansion, deepest-frontier evaluation, and bottom-up reduction schedule, along
with root/history provenance and evaluator. Node IDs and labels remain scoped
to the context UUID and epoch; stale epoch selections fail before output writes.

Runtime identity, writable pointer bounds, device/type/alignment, nonaliasing,
Torch synchronization/owner retention, stream drain, and failure discard rules
are shared with the replay bridge. All five producer/consumer HIP runtime tokens
must match before context creation. HIP allocation intervals cannot prove a
Torch tensor's logical shape/dtype/view or exclusive ownership; the adapter
validates those. A runtime fault can partially write output, requiring discard
and context recreation. Validation/capacity errors remain recoverable.

## Executed comparison

`gpu/teacher_probe.py` built uniform depth3 trees on actual gfx1030 and compared
root and source-ply1 labels with the Rust `cpu-resident` backend. The probe
explicitly adjusts CPU mate scores to the original tree's absolute ply frame.
It uses both classical evaluators on seven fixtures covering Material ties,
white and black mate-in1, forced loss-in2 and winning descendants, EP, promotion,
castles, draw counters, and saturated fullmove metadata.

The executed comparison covered 140 unique parent/evaluator comparisons and
1358 edge rows, including 196 mate-distance rows and 420 tied Material start
rows. It checked exact scores, chosen canonical actions, parent ply, and replay
parent identity. Unscored and unreduced states, invalid schema/reserved/device,
host/null/misaligned/overlapping output, row capacity, and aliased numeric IDs
with stale epochs were rejected.

Twenty-eight marked teacher calls each issued two 16-byte status D2H copies
through `hipMemcpy`/`hipMemcpyAsync`, with no H2D or D2D copies through those APIs.
Diagnostic packet/tensor inspection and the Rust CPU oracle run outside that
scope. The shared replay trace's intentional 128-byte payload controls establish
that these interposers detect payload copies; this is not a complete runtime or
PCIe traffic trace.

The report and raw log are
`gpu/evidence/teacher-gfx1030-2026-09-30.json` and its `.jsonl` companion. Their
source/library/Rust-binary hashes belong to the preserved teacher-only snapshot
at `cpp-rocm:/tmp/chen-teacher-stage`, captured before external callback work.
Later hook source hashes must not be substituted into that evidence. This is
finite correctness evidence, not exhaustive chess verification or measured
playing strength.
