# Codec composition: questions and small experiments

Date: 2026-10-15. This is an algorithm laboratory, not a new chess engine.

## What we are trying to understand

The [original conversation](https://chatgpt.com/share/6ad06b5b-0c50-83eb-94a6-40105b5812af)
proposed horizontal, vertical, and knight representations; interlaced bands and
blocks; precomputed operations; and reducing consequential work. The later
clarification is essential: compression size and incremental updates are not
the objective. Different channels can use different mathematics, while their
composition performs useful work.

The user's current observation is that frequency components can align,
reinforce, or cancel while remaining separately addressable. We should
investigate the algebra behind that observation before designing a decision
procedure. In particular, do not silently replace it with independent
per-move feature extraction followed by a generic classifier.

These are related but distinct mechanisms:

| Mechanism | What the codec actually does | Source |
| --- | --- | --- |
| JPEG DCT | Represents a block using signed coefficients of fixed cosine basis functions. Quantization changes coefficient precision; Huffman coding represents the resulting symbols. | [T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) |
| MDCT / Vorbis | Uses overlapping, critically sampled blocks. Individual inverse blocks contain aliasing terms that cancel during properly windowed overlap-add, before quantization error. | [Princen and Bradley, 1986](https://ieeexplore.ieee.org/document/1164954), [Vorbis specification](https://xiph.org/vorbis/doc/Vorbis_I_spec.html) |
| Opus/CELT | Separates band energy from signed spectral shape; uses Hadamard transforms to adjust time/frequency resolution and supports mid/side stereo. | [RFC 6716, section 4.3](https://datatracker.ietf.org/doc/html/rfc6716#section-4.3) |
| JPEG XR | Composes overlapping spatial transforms hierarchically using reversible integer lifting operations, including sums, differences, and rotation approximations. | [Transform designers' paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/spie2008_hierarchical_lbts.pdf) |
| JPEG 2000 | Uses wavelet decomposition with resolution and quality scalability and selective access. | [JPEG committee overview](https://jpeg.org/jpeg2000/) |
| JPEG XL | Supports variable-sized DCT transforms and other representation tools instead of requiring one fixed block shape throughout. | [Committee white paper](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf) |

The JPEG XL white paper was available as indexed text; direct PDF fetching
timed out during the discussion. It is background, not evidence for an
implemented experiment.

## Useful distinctions

An ordinary DCT has real signed coefficients with fixed basis phases, not an
independently stored arbitrary Fourier phase per coefficient. Different
orthogonal frequencies can cancel at particular locations, but their global
inner product over the complete transform domain is zero. MDCT alias
cancellation is a stronger, deliberately designed relationship between
overlapping blocks; arbitrary averaging of local answers does not inherit it.

For real component values at a selected location,

    (a + b)^2 = a^2 + b^2 + 2ab

Separate energies lose the sign of the cross term. Similarly, mid/side stereo
uses M=(L+R)/2 and S=(L-R)/2: equal inputs occupy the common component;
opposite inputs occupy the difference component. Their separate input
energies can be identical. This is an illustration of retained relative
alignment, not a proposed chess scoring rule.

An invertible transform preserves information. Invertibility alone establishes
neither a better chess algorithm nor lower computational cost. We need to
identify queries that become simpler in the transformed representation.
Conversely, a lossy summary may make the correct answer mathematically
unrecoverable; a pair of inputs with the same summary and different answers
is a useful concrete witness.

## Experimental questions

1. **Alignment:** Can signed/phase-preserving transforms answer a geometrical
   interaction directly, and which discarded components cause ambiguity?
   Does the conclusion survive finite-board boundaries and blocking?
2. **Overlap:** Which cross-boundary relations can local overlapping operators
   recover? When is explicit boundary state necessary? Does reversible
   re-encoding change the answer or merely its representation?
3. **Shared identity:** Can separate target representations compose into an
   exact shared-resource answer? Are counts, energies, or pairwise
   correlations sufficient, including when a particular defender becomes
   unavailable?
4. **Actual alias cancellation:** Reproduce the MDCT mechanism itself, including
   isolated-block alias residuals, and ablate coefficient signs, synthesis
   window agreement, adjacent-block polarity, and quantization. This is a
   signal-processing control, not a chess query.

These questions connect to chess geometry and shared defenders, but their
oracles concern explicitly defined small domains. A matching problem is not a
proof of a winning chess tactic. A geometric attack is not necessarily a legal
move. A current-position relation does not resolve adversarial continuations.

## Round protocol

Each lane owns a separate worktree and directory. Its `protocol.json` fixes the
task, variants, enumeration or seeds, and metrics before outcomes are measured.
There is no model selection, teacher engine, tournament, GPU run, or browser
deployment in this round.

Every lane must retain:

- An independently expressed exact oracle and comparisons of complete outputs.
- Simple controls with the same query contract; deliberate information-loss
  ablations and counterexamples rather than only successful reconstruction.
- Exhaustive small domains where feasible, plus declared larger families if
  needed. Distinguish constructed examples from sampled frequencies.
- Numerical tolerances, representation/table footprint, and any query-versus-
  setup distinction. Timings are exploratory CPU measurements, not Elo or
  deployment performance.
- Small reproducible source, compact results, and an explanation of what the
  outcome establishes and leaves open. Preserve negative results.

The parent agent reruns the frozen lanes and audits important counterexamples.
No aggregate ranking is appropriate across different task contracts.

## Results

Pending at registration. Lane reports will be linked here after execution;
the final comparison must distinguish algebraic guarantees, observed errors,
and architectural hypotheses.
