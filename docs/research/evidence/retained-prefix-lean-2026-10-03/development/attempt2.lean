import proofs.CacheAndBackup

/- Finite mechanical model for retained_canonical_prefix_v1. This imports the
existing cached-winner certificate without re-auditing the old proof archive.
No claim of HIP refinement, hardware latency optimality or chess strength. -/
namespace Chen.Retained

def childOpen (n : Nat) (children : Nat → Nat → Bool) (openLeaves : Nat → Nat)
    (parent child : Nat) : Nat :=
  if child < n ∧ children parent child then openLeaves child else 0

theorem eligible_child_membership (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves reservations : Nat → Nat) (parent child : Nat)
    (available : Eligible (childOpen n children openLeaves parent) reservations child) :
    child < n ∧ children parent child := by
  by_cases valid : child < n ∧ children parent child
  · exact valid
  · simp [Eligible, childOpen, valid] at available

theorem retained_edge_stable (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves old new : Nat → Nat) (scores : Nat → Int) (parent child : Nat)
    (monotone : ∀ node, old node ≤ new node)
    (best : CanonicalBest (Eligible (childOpen n children openLeaves parent) old) scores child)
    (available : Eligible openLeaves new child) :
    CanonicalBest (Eligible (childOpen n children openLeaves parent) new) scores child := by
  have member := eligible_child_membership n children openLeaves old parent child best.1
  apply cached_winner_stable _ old new scores child monotone best
  simpa [Eligible, childOpen, member] using available

def canonicalPath (n : Nat) (children : Nat → Nat → Bool) (openLeaves reservations : Nat → Nat)
    (scores : Nat → Int) : List Nat → Prop
  | [] => True
  | [node] => node < n
  | parent :: child :: rest => parent < n ∧
      CanonicalBest (Eligible (childOpen n children openLeaves parent) reservations) scores child ∧
      canonicalPath n children openLeaves reservations scores (child :: rest)

theorem retained_path_stable (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves old new : Nat → Nat) (scores : Nat → Int) (path : List Nat)
    (monotone : ∀ node, old node ≤ new node)
    (certified : canonicalPath n children openLeaves old scores path)
    (available : ∀ node ∈ path, Eligible openLeaves new node) :
    canonicalPath n children openLeaves new scores path := by
  induction path with
  | nil => trivial
  | cons parent tail ih =>
    cases tail with
    | nil => exact certified
    | cons child rest =>
      exact ⟨certified.1,
        retained_edge_stable n children openLeaves old new scores parent child monotone certified.2.1
          (available child (by simp)),
        ih certified.2.2 (fun node member => available node (by simp [member]))⟩

/- Two certified paths with a common root agree throughout the shorter path.
The reference traversal must extend at least to the retained-retainedPath endpoint;
that termination/rank premise is supplied separately by the finite tree. -/
theorem canonical_prefix_unique (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves reservations : Nat → Nat) (scores : Nat → Int) (retainedPath restart : List Nat)
    (left : canonicalPath n children openLeaves reservations scores retainedPath)
    (right : canonicalPath n children openLeaves reservations scores restart)
    (sameRoot : retainedPath.head? = restart.head?) (lengthBound : retainedPath.length ≤ restart.length) :
    retainedPath = restart.take retainedPath.length := by
  induction retainedPath generalizing restart with
  | nil => simp
  | cons parent tail ih =>
    cases restart with
    | nil => simp at sameRoot
    | cons other remaining =>
      have roots : parent = other := by simpa using sameRoot
      subst other
      cases tail with
      | nil => simp
      | cons child rest =>
        cases remaining with
        | nil => simp at lengthBound
        | cons next suffix =>
          have winners : child = next := canonical_winner_unique _ scores child next left.2.1 right.2.1
          subst next
          have shorter : (child :: rest).length ≤ (child :: suffix).length := by
            simpa using lengthBound
          have tailEq := ih (child :: suffix) left.2.2 right.2.2 (by rfl) shorter
          simpa using congrArg (List.cons parent) tailEq

/- Trimming a suffix preserves the old canonical-edge certificate. -/
theorem canonical_take (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves reservations : Nat → Nat) (scores : Nat → Int) (path : List Nat)
    (certified : canonicalPath n children openLeaves reservations scores path) (count : Nat) :
    canonicalPath n children openLeaves reservations scores (path.take count) := by
  induction path generalizing count with
  | nil => trivial
  | cons parent tail ih =>
    cases count with
    | zero => trivial
    | succ count =>
      cases tail with
      | nil => simpa [canonicalPath] using certified
      | cons child rest =>
        cases count with
        | zero => exact certified.1
        | succ count =>
          exact ⟨certified.1, certified.2.1, ih certified.2.2 (count + 1)⟩

/- An internal parent's open/reservation counts are sums over its children.
All other children contribute nonnegative remaining capacity. This scalar
certificate explains why an eligible tail implies eligible ancestors; deriving
the sum invariant from actual reservations/array writes is not proved here. -/
theorem child_available_parent_available (childOpen childReserved otherOpen otherReserved : Nat)
    (childAvailable : childReserved < childOpen) (othersBound : otherReserved ≤ otherOpen) :
    childReserved + otherReserved < childOpen + otherOpen := by omega

def allAvailable (openLeaves reservations : Nat → Nat) : List Nat → Prop
  | [] => True
  | node :: rest => Eligible openLeaves reservations node ∧ allAvailable openLeaves reservations rest

def tailAvailable (openLeaves reservations : Nat → Nat) : List Nat → Prop
  | [] => True
  | [node] => Eligible openLeaves reservations node
  | _ :: child :: rest => tailAvailable openLeaves reservations (child :: rest)

theorem eligible_tail_covers_ancestors (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves old new : Nat → Nat) (scores : Nat → Int) (path : List Nat)
    (certified : canonicalPath n children openLeaves old scores path)
    (tailEligible : tailAvailable openLeaves new path)
    (upward : ∀ parent child, children parent child →
      Eligible openLeaves new child → Eligible openLeaves new parent) :
    allAvailable openLeaves new path := by
  induction path with
  | nil => trivial
  | cons parent tail ih =>
    cases tail with
    | nil => exact ⟨tailEligible, trivial⟩
    | cons child rest =>
      have tailReady := ih certified.2.2 tailEligible
      have edge := eligible_child_membership n children openLeaves old parent child certified.2.1.1
      exact ⟨upward parent child edge.2 tailReady.1, tailReady⟩

theorem all_available_members (openLeaves reservations : Nat → Nat) (path : List Nat)
    (ready : allAvailable openLeaves reservations path) :
    ∀ node ∈ path, Eligible openLeaves reservations node := by
  induction path with
  | nil => simp
  | cons head tail ih =>
    intro node member
    cases List.mem_cons.mp member with
    | inl same => subst node; exact ready.1
    | inr inside => exact ih ready.2 node inside

theorem retained_restart_equivalence (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves old new : Nat → Nat) (scores : Nat → Int) (oldPath restart : List Nat) (keep : Nat)
    (monotone : ∀ node, old node ≤ new node)
    (certified : canonicalPath n children openLeaves old scores oldPath)
    (tailEligible : tailAvailable openLeaves new (oldPath.take keep))
    (upward : ∀ parent child, children parent child →
      Eligible openLeaves new child → Eligible openLeaves new parent)
    (reference : canonicalPath n children openLeaves new scores restart)
    (sameRoot : (oldPath.take keep).head? = restart.head?)
    (lengthBound : (oldPath.take keep).length ≤ restart.length) :
    oldPath.take keep = restart.take (oldPath.take keep).length := by
  have oldCertificate := canonical_take n children openLeaves old scores oldPath certified keep
  have allReady := eligible_tail_covers_ancestors n children openLeaves old new scores
    (oldPath.take keep) oldCertificate tailEligible upward
  have newCertificate := retained_path_stable n children openLeaves old new scores (oldPath.take keep)
    monotone oldCertificate (all_available_members openLeaves new _ allReady)
  exact canonical_prefix_unique n children openLeaves new scores _ restart
    newCertificate reference sameRoot lengthBound

/- Abstract stack operations account for path pushes and exhausted suffix pops.
Pop at depth0 is invalid. The original root entry is not counted as a push. -/
inductive StackOp where
  | push
  | pop
  deriving DecidableEq

def stackDepth : Nat → List StackOp → Nat
  | depth, [] => depth
  | depth, .push :: rest => stackDepth (depth + 1) rest
  | depth, .pop :: rest => stackDepth (depth - 1) rest

def legalStack : Nat → List StackOp → Prop
  | _, [] => True
  | depth, .push :: rest => legalStack (depth + 1) rest
  | depth, .pop :: rest => 0 < depth ∧ legalStack (depth - 1) rest

def pushes : List StackOp → Nat
  | [] => 0
  | .push :: rest => pushes rest + 1
  | .pop :: rest => pushes rest

def pops : List StackOp → Nat
  | [] => 0
  | .push :: rest => pops rest
  | .pop :: rest => pops rest + 1

theorem stack_accounting (depth : Nat) (operations : List StackOp)
    (legal : legalStack depth operations) :
    stackDepth depth operations + pops operations = depth + pushes operations := by
  induction operations generalizing depth with
  | nil => simp [stackDepth, pops, pushes]
  | cons operation rest ih =>
    cases operation with
    | push =>
      have tail := ih (depth + 1) legal
      simp only [stackDepth, pops, pushes]
      omega
    | pop =>
      have positive := legal.1
      have tail := ih (depth - 1) legal.2
      simp only [stackDepth, pops, pushes]
      omega

theorem pops_bounded_by_pushes (operations : List StackOp) (legal : legalStack 0 operations) :
    pops operations ≤ pushes operations := by
  have accounting := stack_accounting 0 operations legal
  omega

/- Each attempted path has retained edges r and newly discovered suffix edges d.
Root restart calls the cached child helper r+d times; retained traversal calls
it d times. Reservation writes still cover every ancestor in both algorithms. -/
structure Visit where
  retained : Nat
  discovered : Nat

def restartCalls (visits : List Visit) : Nat :=
  (visits.map (fun visit => visit.retained + visit.discovered)).sum

def retainedCalls (visits : List Visit) : Nat :=
  (visits.map (fun visit => visit.discovered)).sum

def savedCalls (visits : List Visit) : Nat :=
  (visits.map (fun visit => visit.retained)).sum

theorem exact_call_accounting (visits : List Visit) :
    restartCalls visits = retainedCalls visits + savedCalls visits := by
  induction visits with
  | nil => simp [restartCalls, retainedCalls, savedCalls]
  | cons visit rest ih =>
    simp only [restartCalls, retainedCalls, savedCalls, List.map_cons, List.sum_cons] at *
    omega

theorem retained_calls_no_more (visits : List Visit) : retainedCalls visits ≤ restartCalls visits := by
  have accounting := exact_call_accounting visits
  omega

/- A conditional integer work model, not measured time: skipped cached-hit work
must exceed new stack overhead. Coefficients and mapping to hardware are premises. -/
theorem weighted_work_improves (visits : List Visit) (hitCost stackOverhead : Nat)
    (benefit : stackOverhead < hitCost * savedCalls visits) :
    hitCost * retainedCalls visits + stackOverhead < hitCost * restartCalls visits := by
  rw [exact_call_accounting, Nat.mul_add]
  omega

end Chen.Retained
