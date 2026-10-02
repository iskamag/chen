import Std

/- Mechanical policy proofs. No axioms, sorry, chess-strength theorem, GPU
memory-model theorem, or correspondence proof to the HIP program is asserted.
Scores/openLeaves metadata are fixed for one choose; reservations only increase.
-/
namespace Chen

def Eligible (openLeaves reservations : Nat → Nat) (child : Nat) : Prop :=
  reservations child < openLeaves child

def CanonicalBest (eligible : Nat → Prop) (score : Nat → Int) (winner : Nat) : Prop :=
  eligible winner ∧ ∀ child, eligible child →
    score child ≤ score winner ∧ (score child = score winner → winner ≤ child)

theorem reservations_shrink_eligibility (openLeaves old new : Nat → Nat)
    (monotone : ∀ child, old child ≤ new child) :
    ∀ child, Eligible openLeaves new child → Eligible openLeaves old child := by
  intro child available
  exact Nat.lt_of_le_of_lt (monotone child) available

theorem canonical_best_under_subset (old new : Nat → Prop) (score : Nat → Int)
    (winner : Nat) (subset : ∀ child, new child → old child)
    (best : CanonicalBest old score winner) (available : new winner) :
    CanonicalBest new score winner := by
  exact ⟨available, fun child valid => best.2 child (subset child valid)⟩

theorem cached_winner_stable (openLeaves old new : Nat → Nat) (score : Nat → Int)
    (winner : Nat) (monotone : ∀ child, old child ≤ new child)
    (best : CanonicalBest (Eligible openLeaves old) score winner)
    (available : Eligible openLeaves new winner) :
    CanonicalBest (Eligible openLeaves new) score winner := by
  exact canonical_best_under_subset _ _ _ _
    (reservations_shrink_eligibility openLeaves old new monotone) best available

theorem canonical_winner_unique (eligible : Nat → Prop) (score : Nat → Int)
    (a b : Nat) (ha : CanonicalBest eligible score a)
    (hb : CanonicalBest eligible score b) : a = b := by
  have ab := ha.2 b hb.1
  have ba := hb.2 a ha.1
  have scores : score a = score b := Int.le_antisymm ba.1 ab.1
  exact Nat.le_antisymm (ab.2 scores.symm) (ba.2 scores)

/- An ancestor-closed affected set cannot have an affected child beneath an
unaffected parent. New sibling values must be initialized identically first.
-/
theorem unaffected_children (childOf : Nat → Nat → Prop) (affected : Nat → Prop)
    (closed : ∀ parent child, childOf parent child → affected child → affected parent)
    (parent : Nat) (unchanged : ¬ affected parent) :
    ∀ child, childOf parent child → ¬ affected child := by
  intro child edge changed
  exact unchanged (closed parent child edge changed)

/- Uniqueness of bounded acyclic value/metadata backup. α can be the entire
score/best-child/openLeaves/min/max record. Both completed orders must satisfy the
same locality reduction equations and agree outside the affected ancestor set.
The equations must depend only on greater-ply children. This proves abstract
order equivalence, not that a particular machine-code pass meets the premises.
-/
theorem backup_order_unique {α : Type} (rank : Nat → Nat) (bound : Nat)
    (rankBound : ∀ node, rank node ≤ bound) (affected : Nat → Prop)
    (reduce : Nat → (Nat → α) → α) (left right : Nat → α)
    (locality : ∀ node x y, (∀ child, rank node < rank child → x child = y child) →
      reduce node x = reduce node y)
    (leftEq : ∀ node, affected node → left node = reduce node left)
    (rightEq : ∀ node, affected node → right node = reduce node right)
    (unchanged : ∀ node, ¬ affected node → left node = right node) :
    ∀ node, left node = right node := by
  have inductionClaim : ∀ height node, rank node + height = bound → left node = right node := by
    intro height
    induction height using Nat.strong_induction_on with
    | h height ih =>
      intro node heightEq
      by_cases changed : affected node
      · rw [leftEq node changed, rightEq node changed]
        apply locality node left right
        intro child deeper
        have childBound := rankBound child
        have smaller : bound - rank child < height := by omega
        have childHeight : rank child + (bound - rank child) = bound := by omega
        exact ih (bound - rank child) smaller child childHeight
      · exact unchanged node changed
  intro node
  apply inductionClaim (bound - rank node) node
  have nodeBound := rankBound node
  omega

/- SWAR key for at most 256 canonical children: biased nonnegative score in
the upper bits and inverted canonical offset in the low eight bits. An invalid
lane uses key0; valid scores use bias≥1. Score conversion/clamping is a caller
premise, not a verified C++ cast.
-/
def stableKey (biasedScore offset : Nat) : Nat := biasedScore * 256 + (255 - offset)

theorem key_score_priority (a b i j : Nat) (hi : i ≤ 255) (hj : j ≤ 255)
    (scores : a < b) : stableKey a i < stableKey b j := by
  unfold stableKey
  omega

theorem key_canonical_tie (score i j : Nat) (hi : i ≤ 255) (hj : j ≤ 255)
    (earlier : i < j) : stableKey score j < stableKey score i := by
  unfold stableKey
  omega

theorem valid_key_above_invalid (score offset : Nat) (valid : 1 ≤ score)
    : 0 < stableKey score offset := by
  unfold stableKey
  omega

theorem wave_max_associative (a b c : Nat) : max (max a b) c = max a (max b c) := by
  exact Nat.max_assoc a b c

theorem wave_max_commutative (a b : Nat) : max a b = max b a := by
  exact Nat.max_comm a b

theorem wave_max_identity (a : Nat) : max a 0 = a := by
  exact Nat.max_zero a

/- Arithmetic packing certificate: low16 reservation count can increase to
1024 without affecting upper fields. This is not bitvector/C++ equivalence.
-/
theorem packed_increment_preserves_fields (fields count : Nat) (limit : count < 1024) :
    (fields * 65536 + count + 1) / 65536 = fields ∧
    (fields * 65536 + count + 1) % 65536 = count + 1 := by
  omega

end Chen
