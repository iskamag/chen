import Std

/- Completed search packets describe one fixed value under identical state,
history, evaluator, horizon and leaf semantics. This is a composition model,
not a proof that HIP search computes that value. -/
namespace Chen.SearchBounds

inductive Result where
  | aborted
  | exact (score : Int)
  | lower (score : Int)
  | upper (score : Int)
  deriving DecidableEq

def Sound (r : Result) (value : Int) : Prop :=
  match r with
  | .aborted => True
  | .exact score => score = value
  | .lower score => score ≤ value
  | .upper score => value ≤ score

def lowerScore : Result → Option Int
  | .exact score | .lower score => some score
  | _ => none

def upperScore : Result → Option Int
  | .exact score | .upper score => some score
  | _ => none

def negate : Result → Result
  | .aborted => .aborted
  | .exact score => .exact (-score)
  | .lower score => .upper (-score)
  | .upper score => .lower (-score)

theorem negate_sound (r : Result) (value : Int) (h : Sound r value) :
    Sound (negate r) (-value) := by
  cases r <;> simp_all [Sound, negate] <;> omega

theorem lower_cutoff (r : Result) (value score beta : Int)
    (sound : Sound r value) (known : lowerScore r = some score)
    (cutoff : beta ≤ score) : beta ≤ value := by
  cases r <;> simp_all [Sound, lowerScore] <;> omega

theorem upper_rejection (r : Result) (value score alpha : Int)
    (sound : Sound r value) (known : upperScore r = some score)
    (reject : score ≤ alpha) : value ≤ alpha := by
  cases r <;> simp_all [Sound, upperScore] <;> omega

theorem aborted_has_no_bounds :
    lowerScore .aborted = none ∧ upperScore .aborted = none := by
  exact ⟨rfl, rfl⟩

/- Nonempty root action values; root is their maximum. The values already use
the parent mover's score frame (child negation must occur before composition). -/
def RootValue (values : List Int) (root : Int) : Prop :=
  root ∈ values ∧ ∀ value ∈ values, value ≤ root

theorem root_lower_from_one_child (values : List Int) (root child score : Int)
    (h : RootValue values root) (member : child ∈ values)
    (bound : score ≤ child) : score ≤ root := by
  have upper := h.2 child member
  omega

theorem root_upper_requires_all_children (values : List Int) (root score : Int)
    (h : RootValue values root) (all : ∀ child ∈ values, child ≤ score) :
    root ≤ score := by
  exact all root h.1

theorem completed_winner (values : List Int) (root candidate : Int)
    (h : RootValue values root) (member : candidate ∈ values)
    (dominates : ∀ child ∈ values, child ≤ candidate) : root = candidate := by
  have lower := h.2 candidate member
  have upper := dominates root h.1
  omega

/- Publication retains a previously completed result when interrupted. Real
iteration publication also has to compare job/state epochs before invoking it. -/
def publish (previous next : Result) : Result :=
  match next with
  | .aborted => previous
  | _ => next

theorem interrupted_publication_preserves (previous : Result) :
    publish previous .aborted = previous := rfl

end Chen.SearchBounds

#print axioms Chen.SearchBounds.negate_sound
#print axioms Chen.SearchBounds.lower_cutoff
#print axioms Chen.SearchBounds.upper_rejection
#print axioms Chen.SearchBounds.aborted_has_no_bounds
#print axioms Chen.SearchBounds.root_lower_from_one_child
#print axioms Chen.SearchBounds.root_upper_requires_all_children
#print axioms Chen.SearchBounds.completed_winner
#print axioms Chen.SearchBounds.interrupted_publication_preserves
