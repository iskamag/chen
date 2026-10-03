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
The reference traversal must extend at least to the retained-prefix endpoint;
that termination/rank premise is supplied separately by the finite tree. -/
theorem canonical_prefix_unique (n : Nat) (children : Nat → Nat → Bool)
    (openLeaves reservations : Nat → Nat) (scores : Nat → Int) (prefix restart : List Nat)
    (left : canonicalPath n children openLeaves reservations scores prefix)
    (right : canonicalPath n children openLeaves reservations scores restart)
    (sameRoot : prefix.head? = restart.head?) (lengthBound : prefix.length ≤ restart.length) :
    prefix = restart.take prefix.length := by
  induction prefix generalizing restart with
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
