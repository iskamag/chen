# Shared identities and joint resource composition

The experiment finds a real interaction that individual channel strengths miss:
**the same resources can appear in several channels, but cannot be spent twice.**
Retaining signed, consistently aligned components permits an exact joint
operator. Merely adding energies, pooling complete vectors, or retaining all
pairwise overlaps can give the wrong collective answer. The useful operation
here is multiplication with an explicit resource-exclusion rule. The Walsh
transform is a change of coordinates for that operation, not the discovery
itself.

This is a finite resource experiment. It establishes no chess legality,
refutation, playing strength, or engine speed claim.

## The query and the interaction

There are four targets and either three or four labelled defenders. An edge
means a defender can serve a target. A query supplies a demanded target subset
and optionally disables one specific defender. Can every demanded target get
a distinct available defender?

Define a variable `x_d` for each defender, and impose `x_d² = 0`. A target's
factor is the sum of its eligible defender variables. Multiplying demanded
target factors enumerates assignments; the square-zero rule removes every
term that spends a defender twice. The sum of surviving coefficients counts
compatible assignments. A positive count answers the joint feasibility query.

For example, compare these two sets of factors:

```text
A: (x0+x1)(x0+x2)(x0+x3)(x0+x1)
B: (x0+x1)(x0+x2)(x1+x2)(x0+x1)
```

Every target has strength two in both cases. The entire target Gram matrix,
including every pairwise intersection, is the same:

```text
2 1 1 2
1 2 1 1
1 1 2 1
2 1 1 2
```

After the first two factors, both products are
`x0x1 + x0x2 + x1x2`. The third factor creates the difference:

```text
A after three: x0x1x2 + x0x1x3 + x0x2x3 + x1x2x3
B after three: 2 x0x1x2
A after four:  2 x0x1x2x3
B after four:  0
```

The measured coefficient-sum stages are `1 → 2 → 3 → 4 → 2` and
`1 → 2 → 3 → 2 → 0`. Both can serve the first three targets; only A can
serve all four. The third-order alignment affects what the fourth channel can
compose with, despite identical pairwise data. For three targets the exact
assignment count is

```text
n0 n1 n2 − c01 n2 − c02 n1 − c12 n0 + 2 c012,
```

where `n` are individual counts and `c` are shared-resource intersections.
Here the triple intersection is one in A and zero in B: `8 − 6 + 2 = 4`
versus `8 − 6 = 2`. This is a concrete missing cross term, rather than a
generic assertion that more information helps.

An earlier failure already occurs with identical target counts **and**
identical defender degrees. Rows below are defenders, columns are targets:

```text
A             B
1100          1010
0011          0101
0010          0010
0001          0001
```

Both have target counts `[1,1,2,2]` and defender degrees `[2,2,1,1]`.
Consequently both counts and the sum of all demanded signed target vectors
coincide. In A the first two targets compete for defender zero; its first two
factors multiply to zero. B has one complete assignment. Keeping total supply
and total demand does not retain which supply can satisfy which demand.

## Exact composition in signed coordinates

Each target incidence vector is zero-padded to four defender slots and encoded
with the unnormalized Sylvester matrix
`H[f,d] = (−1)^popcount(f & d)`: `F_t = H A_t`. This is an exact signed
sum/difference basis. Source labels must agree across all targets.

The joint resource polynomial has `2^n` subset coefficients. Its Walsh state
is `z`; initially all coordinates are one, representing the polynomial one.
Resource creation in this state is

```text
(T_d z)[k] = (−1)^bit_d(k) (z[k] + z[k xor 2^d]) / 2.
```

Combining this with the signed target factor gives the fixed creation tensor

```text
C_f[k,l] = Σ_d H[f,d] (−1)^bit_d(k)
                (δ[l,k] + δ[l,k xor 2^d])
z' = Σ_f F_t[f] C_f z / 8.
```

The implementation keeps integer scale `8^number_of_factors`, so every
comparison is exact. The final DC coordinate is the scaled compatible
assignment count. `compose_signed` performs these updates directly: it never
reconstructs incidence, invokes the permutation oracle, or calls a matching
routine. The oracle independently enumerates injective labelled assignments
from graph bits.

Disabling defender `d` is also an operator in target coordinates:
`F'_t = (I − h_d h_dᵀ/4) F_t`, where `h_d` is column `d` of H. The experiment
checks this against separately encoded available incidence. All checks pass.
This preserves the physical identity of the disabled resource across channels.

## Frozen protocol and actual results

Protocol commit `d630d9a` preceded outcomes; separate alignment intervention
commit `258e97a` preceded execution. Source was frozen at `bf04e89` before the
first measured run. The addendum's conditional-strength clarification was
also committed before that run. No fit, training, or parameter selection was
performed.

All 4,096 three-defender graphs and 65,536 four-defender graphs were enumerated,
with every one of 16 demand subsets and every none/single-removal condition:
262,144 and 5,242,880 query cases respectively. Each representation is grouped
within a fixed known demand and condition. A group requiring both answers is
an information collision; its smaller answer class is an unavoidable error
for any deterministic decoder of that representation.

The table reports **ambiguous cases / minimum possible decoder errors**.
These are information limits, not errors of a fitted model.

| Representation | Three defenders | Four defenders |
|---|---:|---:|
| Per-target counts, energy, or signed DC | 29,664 / 12,672 | 866,064 / 252,418 |
| Two signed components per target | 11,392 / 5,440 | 415,312 / 178,704 |
| Three signed components per target | 0 / 0 | 13,168 / 5,118 |
| Full signed components, aligned identities | 0 / 0 | 0 / 0 |
| Full magnitudes, signs discarded | 27,072 / 12,288 | 667,536 / 227,734 |
| Sum of demanded full signed vectors | 33,888 / 14,592 | 1,295,120 / 422,722 |
| Counts plus demanded signed sum | 1,440 / 288 | 120,216 / 23,952 |
| Full postcondition Gram | 0 / 0 | 288 / 144 |
| Root Gram, labelled condition supplied | 56,448 / 18,816 | 1,458,464 / 462,224 |
| Full signed vectors, independently relabelled per target | 29,664 / 12,672 | 866,064 / 252,418 |
| Full available incidence control | 0 / 0 | 0 / 0 |

All postcondition Gram collisions occur with four available defenders and all
four targets demanded. Thus pairwise overlap was sufficient for feasibility
on the smaller tested domain, but not the larger one. This is an exhaustive
finite result, not a universal sufficiency theorem.

The full signed joint operator agreed with **every exact matching count** on
all 5,505,024 cases, not only its positive/zero classification. Conditional
signed projection had zero coefficient mismatches.

| Actual joint operator | Three defenders FP / FN | Four defenders FP / FN |
|---|---:|---:|
| Full signed | 0 / 0 | 0 / 0 |
| Keep DC, zero-fill missing components | 34,176 / 0 | 458,946 / 0 |
| Keep two, zero-fill missing components | 18,240 / 0 | 360,960 / 0 |
| Keep three, zero-fill missing component | 14,848 / 0 | 350,746 / 0 |
| Interpret magnitudes as signed inputs | 0 / 26,304 | 41,276 / 350,922 |
| Keep three, complete from a known unavailable slot | 0 / 0 | 27,418 / 0 |

Three components retain enough information in the three-defender domain
because the padded fourth incidence is known zero. That constraint determines
the missing component. Blind zero-fill nevertheless fails and even produces
15,104 negative DC outputs. With four defenders, completion becomes exact
under any single-removal condition; its remaining 27,418 false positives are
all in the unconditioned domain where there is no known-zero slot. A
representation and the operator's use of its constraints are separate issues.
The ablated DC values can be fractional after removing the retained integer
scale, or negative. They are diagnostic operator outputs, not matching counts
or probabilities; only the full exact operator has that count interpretation.

## Changing alignment while holding strengths fixed

A coherent global swap of defender labels zero and one, accompanied by the
same swap of any disabled-defender condition, changed **zero matching counts**
throughout both exhaustive domains. This changes coordinates only.

Swapping those defenders in **target zero alone** changes actual resource
sharing. With no condition, it preserves every individual target count and
energy, yet changes feasibility in 2,496 / 65,536 three-defender queries and
17,440 / 1,048,576 four-defender queries. Matching counts change in 5,568 and
119,296 queries respectively. The smallest witness is two targets each
supported only by defender zero versus one supported by zero and the other
by one. Both channels still have strength one, but the joint answer changes
from impossible to possible.

Under a fixed removal of defender zero or one, this local swap can also change
the **available** target strength: it does so on 2,048 / 4,096 and
32,768 / 65,536 graphs respectively. Those conditional results are not
same-available-strength interventions. Removal of another defender preserves
available strengths; the raw output separates every condition. Root Gram's
conditional failures likewise show missing labelled identity, not an
unchanged postcondition Gram.

## Limits and what to carry forward

The result supports composing channels through a shared resource algebra:
retain identities, preserve alternative compatible allocations, and enforce
exclusion in the operator. Signed shape matters because relative components
locate shared resources; energy alone cannot do that. Independent channel
relabeling destroys the alignment even when every local vector is complete.
Pairwise correlations miss a demonstrated higher-order interaction.

There is no demonstrated computational shortcut here. The actual joint state
has 8 or 16 coordinates, and its size grows as `2^n`. The dense creation
tensors have 256 or 1,024 entries (128 or 296 nonzero). Target signed inputs
could occupy 16 int8 bytes in this domain; packed incidence needs only two
bytes. The actual experiment uses int64 arrays: its tensors occupy 2,048 or
8,192 bytes, rather than their compact-int8 sizes recorded in the JSON.
These sizes are bookkeeping, not an optimization objective. The first full
run took 41.243 seconds on CPU 1, mostly grouping and ablations; batch
permutation-oracle time was only 0.0605 seconds. The dense transformed
operator establishes direct composition, not a speed advantage over the
ordinary finite control.

For a next experiment, a useful falsifiable question is whether small,
overlapping *local* resource algebras and messages between them preserve the
important incompatibilities without retaining a global subset state. They
must be tested on the Gram-equal four-target witness above and larger
identity-sharing adversaries; success on isolated channel strengths cannot
answer that question. Turning geometric chess support into this capacity
model would itself need justification. No legal chess refutation has been
shown to disappear through attack/defense cancellation here.

## Reproduce

From the repository/worktree root:

```sh
/tmp/chen-jpeg2-cpu-venv/bin/python experiments/codec_lab/identity/run.py \
  --cpu 1 --out /tmp/chen-identity-rerun.json
```

Only NumPy is required. The recorded environment is Python 3.14.8, NumPy
2.5.4, affinity `[1]`. The committed 688 KB JSON contains all per-condition,
per-demand records, deterministic collisions, operator ablations, worked
traces, timings, and source/protocol hashes. Timing fields vary on rerun.

The parent independently replayed the complete frozen source in 42.071
seconds. All deterministic fields matched after stripping only `seconds`,
`oracle_seconds`, and `total_seconds`; canonical SHA-256 was
`c0ce1d7624e3d9e420d89ea260fdb1b96cd278968c039530f51d9639ff46762c`.
Its independent subset-coefficient dynamic program also agreed with all
query/condition results on its sampled graphs. Astra's independent source
audit checked the `/8` tensor normalization, state retention, projection,
and completion algebra. Those checks supplement the complete permutation
oracle comparison in this lane.

Frozen source SHA-256:
`6cc082833c0274097be6a11fdca326a3dd80fad371a1af41997beb83ec9986e5`.
Protocol SHA-256:
`b7f909b9831374102dc42200caacb77ac8f8e94f9978dde1304b8f8400322447`.
Alignment-addendum SHA-256:
`056fbd1e2ac468dd770964317877e957cd84247c23e6ce181f884de96703b6c4`.
