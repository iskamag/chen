# Fixed-state interaction depth

This experiment asks whether repeated composition resolves higher-order resource compatibility while **the full input and the latent-state coordinates remain fixed**. It does not receive another target or edge each round. It contains no chess positions, trained model, move-tree search, assignment-path objects, or dynamically growing frontier.

The answer is yes for this deliberately exact, exponential algebra. That establishes an interaction-depth effect, not scalable reasoning or a new matching algorithm.

## Fixed input and operation

Every target/resource incidence edge is present at round zero. The fixed Boolean tensor has coordinates `(used target subset, used resource subset)`. Its initial nonzero entries are the unit and all singleton edges. Multiple assignments with the same two subsets collapse to the same coordinate.

Each round computes

`A_next = A OR disjoint_product(A, A)`.

A product is retained only if both target subsets and resource subsets are disjoint. It unions the two target subsets and the two resource subsets. In polynomial notation both repeated target variables and repeated resource variables vanish. Boolean coefficients record existence, not counts. No matching oracle enters this update.

All pair indices are precomputed from dimensions, independently of input edges. Each round traverses this fixed table and writes the same-shaped tensor. This is a dense finite algebraic circuit over subset states, not an implicit claim that the underlying combinatorics disappeared.

Interaction order can double each round: singleton edges at depth0, pairs at depth1, up to four-way combinations at depth2, up to eight-way combinations at depth3. Any nonzero full-target term certifies a compatible assignment. An unchanged complete update certifies a fixed point; without a full-target term that establishes impossibility. Before that, a negative readout is **unknown**, not impossible.

There is also a mathematical completeness bound: every matching of size at most `2^depth` is represented, by recursively partitioning its edges into two compatible sets. With at most six targets, depth3 already represents every matching. We deliberately use the registered, conservative fixed-point readout. Thus depth4's additional negative certificates confirm saturation; they are **not additional matching facts or greater combinational reach**. All observed maximum matching sizes were already exact at depth3.

## Frozen protocol and corpus

Protocol commits `baaee13` and `8de45be` preceded source commit `241c844`; all preceded execution. The amendment bound the supplied schema and included structural cases before any outcomes were known.

Shared corpus SHA-256:

`441a37e200d9460403e95b9318a9d4751790c13d4813e6f51cb3dcf5568311ab`

Selection: all144 structural cases with both dimensions at most6, 16 hash-selected EVAL cases per dimension pair (64 total), and two declared Gram-equal worked cases. Selection uses graph id, not outcomes. No TRAIN or DEV rows are used. This is not a learned generalization test.

For every selected graph, an independent sequential target/resource-mask dynamic program computed maximum matching, including the option to skip a target. It agreed with both supplied `maximum_matching` and `feasible` labels in all210 cases. The shared corpus was separately labelled by the parent's Hall-deficiency oracle. The latent operator never reads those labels. A complete second run reproduced deterministic results byte-for-byte.

## Results

All210 cases, with the fixed full input available from the start:

| Depth | Correct feasible | Correct impossible | Unknown | Incorrect |
|---:|---:|---:|---:|---:|
|0|0|0|210|0|
|1|5|3|202|0|
|2|70|25|115|0|
|3|112|70|28|0|
|4|112|98|0|0|

The64 EVAL cases alone had64/63/34/4/0 unknowns at depths0/1/2/3/4, and zero errors. Structural and worked subsets are separately reported in the JSON; the aggregate is not an independently sampled population estimate.

Frozen-state and additive-reinjection controls remain unknown on all210 cases at every depth. The additive control reinjects the same initial edges, so with Boolean union it is algebraically identical to freezing; these are not two independent pieces of evidence. Their unchanged state does not certify impossibility for the true interaction operator.

Removing resource exclusion while retaining target exclusion gives77 false feasible claims by depth3, persisting at depth4. More rounds therefore amplify an incorrect interaction law; depth alone is not enough. These are deliberately unsound **claimed** positives, not certificates. Negative saturation remains sound for this relaxed superset of the true assignments.

## Worked interaction

Consider four targets with resource choices

- Star: `{a,b}`, `{a,c}`, `{a,d}`, `{a,b}`.
- Triangle: `{a,b}`, `{a,c}`, `{b,c}`, `{a,b}`.

Every target has two choices, and both systems have identical pairwise target-overlap matrices. At depth0 both have9 active coordinates; at depth1 both have25. At depth2 the star has36 coordinates and covers all four targets. The triangle has29 coordinates and covers at most three. The star is immediately certified feasible; the triangle remains unknown until depth3 observes no further change and certifies impossible.

The inputs were not extended between these steps. The distinction comes from composing compatibility among the same edges. This is the practical meaning of interaction depth in this experiment.

## State and cost

For six targets and six resources, one Boolean state occupies4096 bytes. The correct fixed multiplication table contains531,441 pairs and3,188,646 bytes of uint16 indices. The relaxed control contains2,985,984 pairs and17,915,904 index bytes. These are explicit array sizes, not total process memory: initial/current/next states, intermediate Boolean products, tables for other dimensions, and Python/NumPy allocations add overhead.

State size is `2^(T+D)`; the disjoint pair table has `3^(T+D)` entries. The relaxed control has `3^T * 4^D` entries. These exponential costs are a central limitation. The implementation is a small exact subset algebra, not evidence of scalability to chess.

Exploratory CPU1 update medians were roughly95–104 microseconds across the mixed-size corpus; larger instances dominate total time. `timing.json` records one update per instance/depth, excluding oracle, index construction and readout. It is not a stable kernel benchmark, a comparison with optimized matching, or a matched-obligation speed claim. No Walsh version was added: changing coordinates would obscure the single interaction-depth question.

## Reproduce

```sh
taskset -c 1 /tmp/chen-jpeg2-cpu-venv/bin/python \
  experiments/codec_lab/latent_depth/algebra/run.py \
  --corpus /home/iskam/src/vibe/chen/experiments/codec_lab/latent_depth/corpus.json \
  --out /tmp/latent-algebra-results.json \
  --timing-out /tmp/latent-algebra-timing.json
cmp /tmp/latent-algebra-results.json experiments/codec_lab/latent_depth/algebra/results.json
```

The shared corpus is not duplicated in this lane. Results bind exact source, protocol and corpus hashes. No full-engine, learning, novelty, or playing-strength claim follows from this experiment.
