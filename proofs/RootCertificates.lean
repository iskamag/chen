import proofs.SearchBounds

/- Finite root-child packet model. Identity tokens denote equality of the full
objects named, not probabilistic hash equality. Soundness of device workers,
legal-child completeness, and race-free publication remain external premises. -/
namespace Chen.RootCertificates
open Chen.SearchBounds

structure Provenance where
  state : Nat
  history : Nat
  evaluator : Nat
  horizon : Nat
  leafPolicy : Nat
  mateFrame : Nat
  epoch : Nat
  deriving DecidableEq

structure Packet (n : Nat) where
  provenance : Provenance
  child : Fin n
  result : Result

def Admitted {n : Nat} (family : Fin n → Provenance) (values : Fin n → Int)
    (packets : Fin n → Packet n) : Prop :=
  ∀ child, (packets child).provenance = family child ∧
    (packets child).child = child ∧ Sound (packets child).result (-values child)

def parentResult {n : Nat} (packet : Packet n) : Result := negate packet.result

theorem admitted_parent_sound {n : Nat} (family : Fin n → Provenance)
    (values : Fin n → Int) (packets : Fin n → Packet n)
    (admitted : Admitted family values packets) (child : Fin n) :
    Sound (parentResult (packets child)) (values child) := by
  have h := negate_sound _ _ (admitted child).2.2
  simpa [parentResult] using h

theorem wrong_provenance_not_admitted {n : Nat} (family : Fin n → Provenance)
    (values : Fin n → Int) (packets : Fin n → Packet n) (child : Fin n)
    (wrong : (packets child).provenance ≠ family child) :
    ¬ Admitted family values packets := by
  intro admitted
  exact wrong (admitted child).1

theorem child_lower_is_parent_upper (score : Int) :
    negate (.lower score) = .upper (-score) := rfl

theorem child_upper_is_parent_lower (score : Int) :
    negate (.upper score) = .lower (-score) := rfl

theorem lower_endpoint_negates_to_upper (result : Result) (score : Int)
    (known : lowerScore result = some score) : upperScore (negate result) = some (-score) := by
  cases result <;> simp_all [lowerScore, upperScore, negate]

def ChallengerWindowAdmitted (score : Int) : Prop :=
  -32000 < 1 - score ∧ 1 - score ≤ 32000

theorem challenger_window_exact_range (score : Int) :
    ChallengerWindowAdmitted score ↔ -31999 ≤ score ∧ score ≤ 32000 := by
  unfold ChallengerWindowAdmitted
  constructor
  · intro bounds
    constructor <;> omega
  · intro bounds
    constructor <;> omega

theorem mate_range_admits_challenger_window (score : Int)
    (range : -30000 ≤ score ∧ score ≤ 30000) : ChallengerWindowAdmitted score := by
  rw [challenger_window_exact_range]
  constructor <;> omega

theorem minimum_endpoint_needs_other_window : ¬ ChallengerWindowAdmitted (-32000) := by
  intro admitted
  have upper := admitted.2
  omega

/- Proposed challenger child window [-32000, 1-incumbentScore]. A completed
fail-high supplies this lower endpoint; integer arithmetic strictly rejects
the challenger in the parent's frame. No assumption about worker speed. -/
theorem integer_challenger_cutoff_strict (result : Result) (value score endpoint : Int)
    (_window : ChallengerWindowAdmitted score)
    (sound : Sound result (-value)) (known : lowerScore result = some endpoint)
    (cutoff : 1 - score ≤ endpoint) : value < score := by
  have bound := lower_cutoff result (-value) endpoint (1 - score) sound known cutoff
  omega

theorem aborted_parent_has_no_bound {n : Nat} (packet : Packet n)
    (aborted : packet.result = .aborted) :
    lowerScore (parentResult packet) = none ∧ upperScore (parentResult packet) = none := by
  simpa [parentResult, aborted, negate] using aborted_has_no_bounds

def Dominates {n : Nat} (incumbent child : Fin n) (score upper : Int) : Prop :=
  upper ≤ score ∧ (child.val < incumbent.val → upper < score)

def Certificate {n : Nat} (packets : Fin n → Packet n) (incumbent : Fin n)
    (score : Int) : Prop :=
  parentResult (packets incumbent) = .exact score ∧
    ∀ child, child ≠ incumbent → ∃ upper,
      upperScore (parentResult (packets child)) = some upper ∧
      Dominates incumbent child score upper

def CanonicalWinner {n : Nat} (values : Fin n → Int) (incumbent : Fin n) : Prop :=
  (∀ child, values child ≤ values incumbent) ∧
    (∀ child, child.val < incumbent.val → values child < values incumbent)

theorem strict_upper_dominates {n : Nat} (incumbent child : Fin n) (score upper : Int)
    (strict : upper < score) : Dominates incumbent child score upper := by
  exact ⟨by omega, fun _ => strict⟩

theorem certificate_incumbent_exact {n : Nat} (family : Fin n → Provenance)
    (values : Fin n → Int) (packets : Fin n → Packet n) (incumbent : Fin n)
    (score : Int) (admitted : Admitted family values packets)
    (certificate : Certificate packets incumbent score) : values incumbent = score := by
  have sound := admitted_parent_sound family values packets admitted incumbent
  rw [certificate.1] at sound
  exact sound.symm

theorem certificate_canonical_winner {n : Nat} (family : Fin n → Provenance)
    (values : Fin n → Int) (packets : Fin n → Packet n) (incumbent : Fin n)
    (score : Int) (admitted : Admitted family values packets)
    (certificate : Certificate packets incumbent score) : CanonicalWinner values incumbent := by
  have exactValue := certificate_incumbent_exact family values packets incumbent score admitted certificate
  constructor
  · intro child
    by_cases same : child = incumbent
    · subst child; exact Int.le_refl _
    · obtain ⟨upper, known, dominates⟩ := certificate.2 child same
      have bound := upper_rejection _ _ upper upper
        (admitted_parent_sound family values packets admitted child) known (Int.le_refl _)
      have weak := dominates.1
      omega
  · intro child earlier
    have different : child ≠ incumbent := by intro same; subst child; omega
    obtain ⟨upper, known, dominates⟩ := certificate.2 child different
    have bound := upper_rejection _ _ upper upper
      (admitted_parent_sound family values packets admitted child) known (Int.le_refl _)
    have strict := dominates.2 earlier
    omega

theorem certificate_all_children_maximum {n : Nat} (family : Fin n → Provenance)
    (values : Fin n → Int) (packets : Fin n → Packet n) (incumbent : Fin n)
    (score : Int) (admitted : Admitted family values packets)
    (certificate : Certificate packets incumbent score) :
    values incumbent = score ∧ ∀ child, values child ≤ score := by
  have exactValue := certificate_incumbent_exact family values packets incumbent score admitted certificate
  have winner := certificate_canonical_winner family values packets incumbent score admitted certificate
  exact ⟨exactValue, fun child => by simpa [exactValue] using winner.1 child⟩

theorem aborted_competitor_prevents_certificate {n : Nat} (packets : Fin n → Packet n)
    (incumbent child : Fin n) (score : Int) (different : child ≠ incumbent)
    (aborted : (packets child).result = .aborted) : ¬ Certificate packets incumbent score := by
  intro certificate
  obtain ⟨upper, known, _⟩ := certificate.2 child different
  rw [(aborted_parent_has_no_bound (packets child) aborted).2] at known
  contradiction

/- Upper tightening is an information relation, not an admission shortcut.
Both packet sets must independently refer to the SAME family/value function. -/
def TightensUpper {n : Nat} (old new : Fin n → Packet n) : Prop :=
  ∀ child upper, upperScore (parentResult (old child)) = some upper →
    ∃ tighter, upperScore (parentResult (new child)) = some tighter ∧ tighter ≤ upper

theorem domination_tightens {n : Nat} (incumbent child : Fin n) (score upper tighter : Int)
    (dominates : Dominates incumbent child score upper) (tightens : tighter ≤ upper) :
    Dominates incumbent child score tighter := by
  constructor
  · have weak := dominates.1; omega
  · intro earlier; have strict := dominates.2 earlier; omega

theorem certificate_retained_under_tightening {n : Nat} (old new : Fin n → Packet n)
    (incumbent : Fin n) (score : Int) (certificate : Certificate old incumbent score)
    (exactRetained : parentResult (new incumbent) = .exact score)
    (tightens : TightensUpper old new) : Certificate new incumbent score := by
  refine ⟨exactRetained, ?_⟩
  intro child different
  obtain ⟨upper, known, dominates⟩ := certificate.2 child different
  obtain ⟨tighter, newKnown, smaller⟩ := tightens child upper known
  exact ⟨tighter, newKnown, domination_tightens incumbent child score upper tighter dominates smaller⟩

theorem same_family_tightening_preserves_winner {n : Nat} (family : Fin n → Provenance)
    (values : Fin n → Int) (old new : Fin n → Packet n) (incumbent : Fin n)
    (score : Int) (_oldAdmitted : Admitted family values old)
    (newAdmitted : Admitted family values new) (certificate : Certificate old incumbent score)
    (exactRetained : parentResult (new incumbent) = .exact score)
    (tightens : TightensUpper old new) : CanonicalWinner values incumbent := by
  exact certificate_canonical_winner family values new incumbent score newAdmitted
    (certificate_retained_under_tightening old new incumbent score certificate exactRetained tightens)

theorem abort_retains_completed_packet (previous : Result) : publish previous .aborted = previous :=
  interrupted_publication_preserves previous

/- Independent interval oracle model: every integer vector within endpoints is
possible. These adversarial witnesses do not assert independence of chess moves. -/
def Feasible {n : Nat} (lower upper values : Fin n → Int) : Prop :=
  ∀ child, lower child ≤ values child ∧ values child ≤ upper child

def spike {n : Nat} (values upper : Fin n → Int) (challenger : Fin n) : Fin n → Int :=
  fun child => if child = challenger then upper child else values child

theorem spike_feasible {n : Nat} (lower upper values : Fin n → Int) (challenger : Fin n)
    (feasible : Feasible lower upper values) : Feasible lower upper (spike values upper challenger) := by
  intro child
  by_cases same : child = challenger
  · subst child
    have bounds := feasible challenger
    simp [spike]
    omega
  · simpa [spike, same] using feasible child

theorem unresolved_competitor_obstruction {n : Nat} (lower upper values : Fin n → Int)
    (incumbent challenger : Fin n) (feasible : Feasible lower upper values)
    (different : challenger ≠ incumbent) (canExceed : values incumbent < upper challenger) :
    ∃ alternative, Feasible lower upper alternative ∧
      alternative incumbent = values incumbent ∧ ¬ CanonicalWinner alternative incumbent := by
  refine ⟨spike values upper challenger, spike_feasible lower upper values challenger feasible, ?_, ?_⟩
  · simp [spike, Ne.symm different]
  · intro winner
    have domination := winner.1 challenger
    simp [spike, Ne.symm different] at domination
    omega

theorem earlier_tie_obstruction {n : Nat} (lower upper values : Fin n → Int)
    (incumbent challenger : Fin n) (feasible : Feasible lower upper values)
    (earlier : challenger.val < incumbent.val) (canTie : values incumbent ≤ upper challenger) :
    ∃ alternative, Feasible lower upper alternative ∧
      alternative incumbent = values incumbent ∧ ¬ CanonicalWinner alternative incumbent := by
  have different : challenger ≠ incumbent := by intro same; subst challenger; omega
  refine ⟨spike values upper challenger, spike_feasible lower upper values challenger feasible, ?_, ?_⟩
  · simp [spike, Ne.symm different]
  · intro winner
    have strict := winner.2 challenger earlier
    simp [spike, Ne.symm different] at strict
    omega

def UniversallySafe {n : Nat} (lower upper : Fin n → Int) (incumbent : Fin n) : Prop :=
  ∀ values, Feasible lower upper values → CanonicalWinner values incumbent

theorem unresolved_not_universally_safe {n : Nat} (lower upper values : Fin n → Int)
    (incumbent challenger : Fin n) (feasible : Feasible lower upper values)
    (different : challenger ≠ incumbent) (canExceed : values incumbent < upper challenger) :
    ¬ UniversallySafe lower upper incumbent := by
  obtain ⟨alternative, possible, _, notWinner⟩ :=
    unresolved_competitor_obstruction lower upper values incumbent challenger feasible different canExceed
  intro safe
  exact notWinner (safe alternative possible)

end Chen.RootCertificates
