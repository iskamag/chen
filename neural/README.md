# Neural laboratory

Independent PyTorch experiments live on `research/neural`. The engine continues
to own exact legality and terminal rules. The models expose side-to-move latent
encoding, learned move dynamics, explicit latent branching/rollouts, and shared
policy/WDL/piece/rule decoders. Architecture rationale and prior art are in
[NEURAL_ARCHITECTURES.md](../docs/NEURAL_ARCHITECTURES.md).

Requires PyTorch; replay generation additionally uses python-chess. The existing
`cpp-rocm` container supplies ROCm PyTorch and the gfx1030 device. Run from the
worktree root, or copy this package into an isolated container project directory:

```sh
python -m unittest discover -s neural/tests -v
python -m neural.train --help
python -m neural.train --device cuda --models square relation plan \
  --samples 192 --steps 8 --batch-size 16 --width 32 --depth 1 \
  --output results/neural-smoke.json
```

```python
import torch
from neural.features import ACTION_COUNT, encode_fen, move_to_index
from neural.models import build_model

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
model = build_model("plan", width=64, depth=2).cuda()
features = encode_fen(fen).unsqueeze(0).cuda()
z = model.encode(features)
# These particular moves are legal in this exact root. Subsequent masks and
# moves must come from actual successor boards, not predicted board occupancy.
actions = torch.tensor([[move_to_index("e2e4"), move_to_index("d2d4")]], device="cuda")
branches = model.branch(z, actions)  # [1,2,64,64], all resident
predictions = model.decode(branches.flatten(0, 1))
```

`forward(features, legal_mask=None)` and `decode` return `latent [B,64,D]`,
`wdl_logits [B,3]` (loss/draw/win), `policy_logits [B,20480]`,
`board_logits [B,12,8,8]`, `castling_logits [B,4]`, `ep_logits [B,65]`
(class64 is absent), and `halfmove [B]`. `decode_board` and `decode_rules` avoid
materializing unused policy logits in successor training.

`legal_probabilities(logits, mask)` gives zeros for illegal actions and for
terminal rows. Never interpret unmasked policy logits as legal-move generation.
Actions are int64 and use the schema's current mover orientation. The caller
must supply in-range indices and legal moves; this avoids device synchronization
for host-side per-action validation during training.

The test suite covers schema color/rank symmetry and clocks, all action-label
round trips, blockers and pawn pseudoattacks, terminal-mask normalization,
representation and output gradients, multistep dynamics, resident branching,
and float32 CPU/GPU parity. It does not prove dynamics correctness after training,
neural engine integration, or playing strength. Float32 CPU/GPU comparison uses
absolute/relative tolerance `3e-4`, separately from any future mixed precision.

Generated-data smoke training uses explicitly labeled immediate-move/material
proxies and successor reconstruction. Those labels exercise the optimizer and
controlled experiment machinery; they are not real game outcome supervision.
Each comparison records environment, corpus fingerprint, parameter counts,
sample-sequence fingerprint and loss components. Width-matched comparisons are
not parameter-matched. Timing includes initial optimizer-state allocation and
cannot establish steady-state throughput. No present checkpoint has a supported
Elo claim.
