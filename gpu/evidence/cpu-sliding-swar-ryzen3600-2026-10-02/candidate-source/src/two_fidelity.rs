//! CPU reference for 2FFS (arXiv:2606.01708v2, Appendix B.3).
//!
//! Values use one fixed root-player perspective. This is not a chess evaluator:
//! a deterministic fast error envelope and mean-correct independent slow oracle
//! are required for the paper's conditional guarantee. A rollout is not generally
//! an unbiased sample of minimax value. See `docs/TWO_FIDELITY.md`.
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

pub type NodeId = usize;

/// A finite balanced tree, rooted at 0. Player alternates Max/Min by depth.
/// Children must be in a fixed deterministic order; transpositions are not trees.
pub trait FiniteTree {
    fn node_count(&self) -> usize;
    fn children(&self, node: NodeId) -> &[NodeId];
}

/// Fast queries are deterministic. Implementations can batch a whole expansion
/// on CPU or GPU; results must retain request order and contain every child.
pub trait FastOracle {
    fn evaluate(&mut self, node: NodeId) -> Result<f64, String>;
    fn evaluate_batch(&mut self, nodes: &[NodeId]) -> Result<Vec<f64>, String> {
        nodes.iter().map(|&n| self.evaluate(n)).collect()
    }
}

/// Samples at *any exposed non-root node*, including expanded internal nodes.
/// Independence, sub-Gaussianity, and the declared mean/bias are caller premises.
pub trait SlowOracle {
    fn sample(&mut self, node: NodeId) -> Result<f64, String>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}
impl Interval {
    const UNBOUNDED: Self = Self {
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    };
    pub fn width(self) -> f64 {
        self.upper - self.lower
    }
    fn intersect(self, other: Self) -> Self {
        Self {
            lower: self.lower.max(other.lower),
            upper: self.upper.min(other.upper),
        }
    }
    fn nonempty(self) -> bool {
        self.lower <= self.upper
    }
}

/// A declaration, not runtime proof of stochastic oracle properties.
#[derive(Clone, Debug)]
pub struct OracleContract {
    /// B[h], nondecreasing, with B[0] = 0. Leaves must be fast-exact.
    pub envelope: Vec<f64>,
    pub sigma: f64,
    /// Theorem 3.7 inflation; zero is the unbiased setup.
    pub slow_bias_bound: f64,
    /// Must identify how bounds and stochastic premises were established.
    pub provenance: String,
    /// False for heuristic chess adapters without justified bounds.
    pub justified: bool,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub delta: f64,
    pub epsilon: f64,
    pub slow_cost: f64,
    pub max_rounds: u64,
    pub max_cost: f64,
    /// Floating-point/resource guard, not an alternative stopping condition.
    pub max_scale: usize,
    pub max_nodes: usize,
    pub trace: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            delta: 0.05,
            epsilon: 0.0,
            slow_cost: 2.0,
            max_rounds: 200_000,
            max_cost: 1_000_000.0,
            max_scale: 128,
            max_nodes: 1_000_000,
            trace: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Lower,
    Upper,
}
impl Side {
    fn index(self) -> usize {
        if self == Self::Lower { 0 } else { 1 }
    }
    fn opposite(self) -> Self {
        if self == Self::Lower {
            Self::Upper
        } else {
            Self::Lower
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Separated,
    ConfidenceFailure,
    CostBudget,
    RoundBudget,
    PrecisionLimit,
}
#[derive(Clone, Debug)]
pub struct ActionTrace {
    pub node: NodeId,
    pub side: Side,
    pub scale: usize,
    pub expansion: bool,
    pub cost: f64,
}
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub fast_queries: u64,
    pub slow_queries: u64,
    pub expansions: u64,
    pub refresh_node_visits: u64,
    pub refresh_topology_scans: u64,
    pub certificate_node_visits: u64,
    pub certificate_topology_scans: u64,
    pub scale_storage_scans: u64,
    pub scale_evaluations: u64,
    pub rounds: u64,
    pub fast_time: Duration,
    pub slow_time: Duration,
    pub refresh_time: Duration,
    pub scale_time: Duration,
    pub total_time: Duration,
}
#[derive(Clone, Debug)]
pub struct ResultReport {
    /// Only Separated is an ordinary interval-certified recommendation.
    /// ConfidenceFailure returns root's first child as required by paper B.3.
    pub recommendation: Option<NodeId>,
    pub stop: StopReason,
    /// Records a caller declaration, not proof of stochastic or numeric premises.
    pub oracle_contract_justified: bool,
    pub total_cost: f64,
    pub root_intervals: Vec<(NodeId, Interval)>,
    pub profile: Profile,
    pub trace: Vec<ActionTrace>,
}
#[derive(Clone, Debug)]
struct State {
    exposed: bool,
    expanded: bool,
    fast: Interval,
    slow: Interval,
    local: Interval,
    effective: Interval,
    samples: u64,
    mean: f64,
}
impl Default for State {
    fn default() -> Self {
        Self {
            exposed: false,
            expanded: false,
            fast: Interval::UNBOUNDED,
            slow: Interval::UNBOUNDED,
            local: Interval::UNBOUNDED,
            effective: Interval::UNBOUNDED,
            samples: 0,
            mean: 0.0,
        }
    }
}
#[derive(Clone, Debug, Default)]
struct ScaleState {
    local_cost: f64,
    recursive_cost: f64,
    done: [bool; 2],
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Status {
    Cert,
    Progress,
    Blocked,
}

/// Stateful reference controller. Full-tree topology is supplied to establish a
/// finite confidence allocation; oracle values remain queried only on exposure.
pub struct TwoFidelity<T, F, S> {
    tree: T,
    fast: F,
    slow: S,
    config: Config,
    contract: OracleContract,
    state: Vec<State>,
    depth: Vec<usize>,
    bottom_up: Vec<NodeId>,
    certificate_done: Vec<[bool; 2]>,
    max_depth: usize,
    scales: BTreeMap<(NodeId, usize), ScaleState>,
    // Certificates depend on evidence and the same-scale monotone latches,
    // not oracle-cost counters. One bottom-up pass reaches their fixed point.
    // Retain completed scales until the next evidence refresh.
    valid_scales: BTreeSet<usize>,
    rho0: f64,
    node_delta: f64,
    cost: f64,
    profile: Profile,
    trace: Vec<ActionTrace>,
    failed: bool,
    initialized: bool,
}
impl<T: FiniteTree, F: FastOracle, S: SlowOracle> TwoFidelity<T, F, S> {
    pub fn new(
        tree: T,
        fast: F,
        slow: S,
        config: Config,
        contract: OracleContract,
    ) -> Result<Self, String> {
        if !config.delta.is_finite()
            || !(0.0..1.0).contains(&config.delta)
            || config.delta == 0.0
            || !config.epsilon.is_finite()
            || config.epsilon < 0.0
            || !config.slow_cost.is_finite()
            || config.slow_cost < 1.0
            || !config.max_cost.is_finite()
            || config.max_cost < 0.0
            || config.max_scale > 1022
            || config.max_nodes < 3
            || !contract.sigma.is_finite()
            || contract.sigma < 0.0
            || !contract.slow_bias_bound.is_finite()
            || contract.slow_bias_bound < 0.0
            || contract.provenance.trim().is_empty()
        {
            return Err("invalid configuration or missing oracle provenance".into());
        }
        let n = tree.node_count();
        if n < 3 || n > config.max_nodes || tree.children(0).len() < 2 {
            return Err("invalid tree size/root".into());
        }
        let mut depth = vec![usize::MAX; n];
        depth[0] = 0;
        let mut order = vec![0];
        let mut i = 0;
        while i < order.len() {
            let v = order[i];
            i += 1;
            if depth[v] > 128 {
                return Err("tree depth exceeds CPU reference recursion bound".into());
            }
            for &u in tree.children(v) {
                if u >= n || depth[u] != usize::MAX {
                    return Err("cycle, duplicate child, or transposition".into());
                }
                depth[u] = depth[v] + 1;
                order.push(u);
            }
        }
        if order.len() != n {
            return Err("unreachable tree nodes".into());
        }
        let max_depth = *depth.iter().max().unwrap();
        if (0..n).any(|v| tree.children(v).is_empty() != (depth[v] == max_depth)) {
            return Err("paper reference requires balanced depth".into());
        }
        if contract.envelope.len() != max_depth + 1
            || contract.envelope[0] != 0.0
            || contract.envelope.iter().any(|b| !b.is_finite() || *b < 0.0)
            || contract.envelope.windows(2).any(|w| w[0] > w[1])
        {
            return Err("invalid depth envelope".into());
        }
        let node_delta = config.delta / (n - 1) as f64;
        if node_delta == 0.0 {
            return Err("confidence allocation underflow".into());
        }
        Ok(Self {
            tree,
            fast,
            slow,
            config,
            contract,
            state: vec![State::default(); n],
            depth,
            bottom_up: vec![0],
            certificate_done: vec![[false; 2]; n],
            max_depth,
            scales: BTreeMap::new(),
            valid_scales: BTreeSet::new(),
            rho0: 0.0,
            node_delta,
            cost: 0.0,
            profile: Profile::default(),
            trace: Vec::new(),
            failed: false,
            initialized: false,
        })
    }
    pub fn interval(&self, node: NodeId) -> Option<Interval> {
        self.state
            .get(node)
            .filter(|s| s.exposed)
            .map(|s| s.effective)
    }
    pub fn slow_samples(&self, node: NodeId) -> u64 {
        self.state[node].samples
    }
    pub fn is_expanded(&self, node: NodeId) -> bool {
        self.state[node].expanded
    }
    fn rho(&self, k: usize) -> f64 {
        self.rho0 * 2.0_f64.powi(-(k as i32))
    }
    fn is_max(&self, v: NodeId) -> bool {
        self.depth[v].is_multiple_of(2)
    }
    fn comparison(&self, v: NodeId, side: Side) -> bool {
        self.is_max(v) == (side == Side::Lower)
    }
    /// Conservative two-sided union over all sample counts. Unlike the author's
    /// numerical radius this explicitly pays both tails. It is nonincreasing.
    pub fn radius(&self, n: u64) -> f64 {
        if n == 0 {
            return f64::INFINITY;
        }
        self.contract.sigma
            * (2.0 * (8.0_f64.ln() - self.node_delta.ln() + 2.0 * (n as f64).ln()) / n as f64)
                .sqrt()
            + self.contract.slow_bias_bound
    }
    fn required_samples(&self, target: f64) -> Option<u64> {
        if target <= self.contract.slow_bias_bound {
            return None;
        }
        let mut hi = 1_u64;
        while self.radius(hi) > target {
            hi = hi.checked_mul(2)?;
        }
        let mut lo = 1;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.radius(mid) <= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Some(lo)
    }
    fn race_budget(&self, v: NodeId, k: usize) -> f64 {
        let h = self.max_depth - self.depth[v];
        if self.contract.envelope[h] <= self.rho(k) / 4.0 {
            return 0.0;
        }
        let gamma = self
            .required_samples(self.rho(k) / 4.0)
            .map_or(f64::INFINITY, |n| self.config.slow_cost * n as f64);
        ((h + 1) as f64).powi(2) * gamma
    }
    fn expose(&mut self, nodes: &[NodeId]) -> Result<(), String> {
        let t = Instant::now();
        let values = self.fast.evaluate_batch(nodes)?;
        self.profile.fast_time += t.elapsed();
        if values.len() != nodes.len() || values.iter().any(|x| !x.is_finite()) {
            return Err("fast oracle returned wrong batch size or nonfinite value".into());
        }
        for (&v, value) in nodes.iter().zip(values) {
            if self.state[v].exposed {
                return Err("duplicate exposure".into());
            }
            let b = self.contract.envelope[self.max_depth - self.depth[v]];
            let fast = Interval {
                lower: value - b,
                upper: value + b,
            };
            if !fast.lower.is_finite() || !fast.upper.is_finite() || !fast.width().is_finite() {
                return Err("fast interval overflow".into());
            }
            self.state[v].exposed = true;
            self.state[v].fast = fast;
        }
        self.bottom_up.extend_from_slice(nodes);
        self.bottom_up
            .sort_unstable_by(|&x, &y| self.depth[y].cmp(&self.depth[x]).then(x.cmp(&y)));
        self.profile.fast_queries += nodes.len() as u64;
        self.cost += nodes.len() as f64;
        Ok(())
    }
    fn refresh(&mut self) {
        let start = Instant::now();
        self.valid_scales.clear();
        self.profile.refresh_topology_scans += self.bottom_up.len() as u64;
        for &v in &self.bottom_up {
            if !self.state[v].exposed {
                continue;
            }
            self.profile.refresh_node_visits += 1;
            let mut child = Interval::UNBOUNDED;
            if self.state[v].expanded {
                let init = if self.is_max(v) {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                child = Interval {
                    lower: init,
                    upper: init,
                };
                for &u in self.tree.children(v) {
                    let x = self.state[u].effective;
                    if self.is_max(v) {
                        child.lower = child.lower.max(x.lower);
                        child.upper = child.upper.max(x.upper);
                    } else {
                        child.lower = child.lower.min(x.lower);
                        child.upper = child.upper.min(x.upper);
                    }
                }
            }
            let local = self.state[v].fast.intersect(self.state[v].slow);
            let effective = if v == 0 {
                child
            } else {
                local.intersect(child)
            };
            self.state[v].local = local;
            self.state[v].effective = effective;
            if !local.nonempty() || !effective.nonempty() {
                self.failed = true;
                break;
            }
        }
        self.profile.refresh_time += start.elapsed();
    }
    /// Evaluate certificates bottom-up at tolerance, using optional prior latches.
    fn certificates(&mut self, tolerance: f64, k: Option<usize>) {
        self.profile.certificate_topology_scans += self.bottom_up.len() as u64;
        for &v in &self.bottom_up {
            if v == 0 || !self.state[v].exposed {
                continue;
            }
            self.profile.certificate_node_visits += 1;
            for side in [Side::Lower, Side::Upper] {
                let prior = k.is_some_and(|j| {
                    self.scales
                        .get(&(v, j))
                        .is_some_and(|s| s.done[side.index()])
                });
                let width = self.state[v].effective.width() <= tolerance
                    || self.state[v].local.width() <= tolerance;
                let recursive = if self.state[v].expanded {
                    let children = self.tree.children(v);
                    if self.comparison(v, side) {
                        let lambda = if side == Side::Lower {
                            children
                                .iter()
                                .map(|&u| self.state[u].effective.lower)
                                .fold(f64::NEG_INFINITY, f64::max)
                        } else {
                            children
                                .iter()
                                .map(|&u| self.state[u].effective.upper)
                                .fold(f64::INFINITY, f64::min)
                        };
                        children.iter().all(|&u| {
                            let x = self.state[u].effective;
                            (if side == Side::Lower {
                                x.upper <= lambda + tolerance
                            } else {
                                x.lower >= lambda - tolerance
                            }) || self.certificate_done[u][side.index()]
                        })
                    } else {
                        let endpoint = |u: NodeId| {
                            if side == Side::Lower {
                                self.state[u].effective.lower
                            } else {
                                self.state[u].effective.upper
                            }
                        };
                        let best = children.iter().map(|&u| endpoint(u)).fold(
                            if side == Side::Lower {
                                f64::INFINITY
                            } else {
                                f64::NEG_INFINITY
                            },
                            if side == Side::Lower {
                                f64::min
                            } else {
                                f64::max
                            },
                        );
                        children
                            .iter()
                            .filter(|&&u| endpoint(u) == best)
                            .all(|&u| self.certificate_done[u][side.index()])
                    }
                } else {
                    false
                };
                self.certificate_done[v][side.index()] = prior || width || recursive;
            }
        }
    }
    fn evaluate_scale(&mut self, k: usize) {
        if self.valid_scales.contains(&k) {
            return;
        }
        let start = Instant::now();
        self.profile.scale_evaluations += 1;
        self.profile.scale_storage_scans += 2 * (self.bottom_up.len() - 1) as u64;
        for &v in &self.bottom_up {
            if v != 0 {
                self.scales.entry((v, k)).or_default();
            }
        }
        self.certificates(self.rho(k) / 2.0, Some(k));
        for &v in &self.bottom_up {
            if v != 0 {
                self.scales.get_mut(&(v, k)).unwrap().done = self.certificate_done[v];
            }
        }
        self.profile.scale_time += start.elapsed();
        self.valid_scales.insert(k);
    }
    fn scale_comp(&mut self, v: NodeId, side: Side, k: usize) -> bool {
        self.evaluate_scale(k);
        self.scales[&(v, k)].done[side.index()]
    }
    fn refresh_latch(&mut self, k: usize) {
        self.refresh();
        if !self.failed {
            self.evaluate_scale(k);
        }
    }
    fn active_scale(&mut self, v: NodeId, side: Side) -> Result<Option<usize>, String> {
        self.certificates(0.0, None);
        if self.certificate_done[v][side.index()] {
            return Ok(None);
        }
        let width = self.state[v].effective.width();
        for k in 0..=self.config.max_scale {
            let rho = self.rho(k);
            if rho == 0.0 {
                break;
            }
            if rho / 2.0 < width && width <= rho {
                for j in k..=self.config.max_scale {
                    if self.rho(j) == 0.0 {
                        break;
                    }
                    if !self.scale_comp(v, side, j) {
                        return Ok(Some(j));
                    }
                }
                break;
            }
        }
        Err("floating scale resource limit".into())
    }
    fn capped_scale(&mut self, v: NodeId, side: Side, cap: usize) -> Option<usize> {
        (0..=cap).find(|&k| !self.scale_comp(v, side, k))
    }
    fn contender(&mut self, v: NodeId) -> Result<Option<(Side, usize)>, String> {
        let lower = self.active_scale(v, Side::Lower)?;
        let upper = self.active_scale(v, Side::Upper)?;
        Ok(match (lower, upper) {
            (Some(l), Some(u)) if l <= u => Some((Side::Lower, l)),
            (_, Some(u)) => Some((Side::Upper, u)),
            (Some(l), None) => Some((Side::Lower, l)),
            (None, None) => None,
        })
    }
    fn root_obligation(&mut self) -> Result<(NodeId, Side, usize), String> {
        let children = self.tree.children(0);
        let a = *children
            .iter()
            .max_by(|&&x, &&y| {
                self.state[x]
                    .effective
                    .lower
                    .total_cmp(&self.state[y].effective.lower)
                    .then_with(|| y.cmp(&x))
            })
            .unwrap();
        let b = *children
            .iter()
            .filter(|&&u| u != a)
            .max_by(|&&x, &&y| {
                self.state[x]
                    .effective
                    .upper
                    .total_cmp(&self.state[y].effective.upper)
                    .then_with(|| y.cmp(&x))
            })
            .unwrap();
        let ka = self.active_scale(a, Side::Lower)?;
        let contender = self.contender(b)?;
        match (ka, contender) {
            (Some(k), Some((_, j))) if k <= j => Ok((a, Side::Lower, k)),
            (_, Some((s, j))) => Ok((b, s, j)),
            (Some(k), None) => Ok((a, Side::Lower, k)),
            _ => Err("no unresolved root side".into()),
        }
    }
    fn child_obligation(
        &mut self,
        v: NodeId,
        side: Side,
        k: usize,
    ) -> Result<(NodeId, Side, usize), String> {
        self.evaluate_scale(k);
        let children = self.tree.children(v);
        if self.comparison(v, side) {
            let lambda = if side == Side::Lower {
                children
                    .iter()
                    .map(|&u| self.state[u].effective.lower)
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                children
                    .iter()
                    .map(|&u| self.state[u].effective.upper)
                    .fold(f64::INFINITY, f64::min)
            };
            let tolerance = self.rho(k) / 2.0;
            let mut active: Vec<_> = children
                .iter()
                .copied()
                .filter(|&u| {
                    !self.scales[&(u, k)].done[side.index()]
                        && if side == Side::Lower {
                            self.state[u].effective.upper > lambda + tolerance
                        } else {
                            self.state[u].effective.lower < lambda - tolerance
                        }
                })
                .collect();
            active.sort_by(|&x, &y| {
                let ex = if side == Side::Lower {
                    self.state[x].effective.upper
                } else {
                    -self.state[x].effective.lower
                };
                let ey = if side == Side::Lower {
                    self.state[y].effective.upper
                } else {
                    -self.state[y].effective.lower
                };
                ey.total_cmp(&ex).then(x.cmp(&y))
            });
            let u = *active
                .first()
                .ok_or("unresolved comparison without blocker")?;
            if let Some(j) = self.capped_scale(u, side.opposite(), k) {
                return Ok((u, side.opposite(), j));
            }
            return self
                .capped_scale(u, side, k)
                .map(|j| (u, side, j))
                .ok_or("blocker has no capped side".into());
        }
        let endpoint = |u: NodeId| {
            if side == Side::Lower {
                self.state[u].effective.lower
            } else {
                self.state[u].effective.upper
            }
        };
        let best = children.iter().map(|&u| endpoint(u)).fold(
            if side == Side::Lower {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            },
            if side == Side::Lower {
                f64::min
            } else {
                f64::max
            },
        );
        let u = *children
            .iter()
            .filter(|&&u| endpoint(u) == best && !self.scales[&(u, k)].done[side.index()])
            .max_by(|&&x, &&y| {
                self.state[x]
                    .effective
                    .width()
                    .total_cmp(&self.state[y].effective.width())
                    .then_with(|| y.cmp(&x))
            })
            .ok_or("unresolved selector without witness")?;
        self.capped_scale(u, side, k)
            .map(|j| (u, side, j))
            .ok_or("witness has no capped side".into())
    }
    fn local_step(
        &mut self,
        v: NodeId,
        side: Side,
        k: usize,
        cap: f64,
    ) -> Result<(f64, Status), String> {
        let c = self.config.slow_cost;
        if cap < c {
            return Ok((0.0, Status::Blocked));
        }
        let start = Instant::now();
        let y = self.slow.sample(v)?;
        self.profile.slow_time += start.elapsed();
        if !y.is_finite() {
            return Err("slow oracle returned nonfinite value".into());
        }
        let samples = self.state[v]
            .samples
            .checked_add(1)
            .ok_or("sample count overflow")?;
        let mean = self.state[v].mean + (y - self.state[v].mean) / samples as f64;
        let radius = self.radius(samples);
        let interval = Interval {
            lower: mean - radius,
            upper: mean + radius,
        };
        if !mean.is_finite() || interval.lower.is_nan() || interval.upper.is_nan() {
            return Err("sample arithmetic overflow".into());
        }
        self.state[v].samples = samples;
        self.state[v].mean = mean;
        self.state[v].slow = self.state[v].slow.intersect(interval);
        self.scales.entry((v, k)).or_default().local_cost += c;
        self.cost += c;
        self.profile.slow_queries += 1;
        if self.config.trace {
            self.trace.push(ActionTrace {
                node: v,
                side,
                scale: k,
                expansion: false,
                cost: c,
            });
        }
        self.refresh_latch(k);
        Ok((c, Status::Progress))
    }
    fn expand(&mut self, v: NodeId, side: Side, k: usize) -> Result<(f64, Status), String> {
        let children = self.tree.children(v).to_vec();
        let cost = children.len() as f64;
        self.expose(&children)?;
        self.state[v].expanded = true;
        self.scales.entry((v, k)).or_default().recursive_cost += cost;
        self.profile.expansions += 1;
        if self.config.trace {
            self.trace.push(ActionTrace {
                node: v,
                side,
                scale: k,
                expansion: true,
                cost,
            });
        }
        self.refresh_latch(k);
        Ok((cost, Status::Progress))
    }
    fn resolve(
        &mut self,
        v: NodeId,
        side: Side,
        k: usize,
        cap: f64,
    ) -> Result<(f64, Status), String> {
        if self.scale_comp(v, side, k) {
            return Ok((0.0, Status::Cert));
        }
        if self.tree.children(v).is_empty() {
            return self.local_step(v, side, k, cap);
        }
        let remaining = self.race_budget(v, k) - self.scales[&(v, k)].recursive_cost;
        if remaining <= 0.0 {
            return self.local_step(v, side, k, cap);
        }
        if !self.state[v].expanded {
            let cost = self.tree.children(v).len() as f64;
            if cost > remaining {
                return self.local_step(v, side, k, cap);
            }
            if cost > cap {
                return Ok((0.0, Status::Blocked));
            }
            return self.expand(v, side, k);
        }
        let (u, child_side, j) = self.child_obligation(v, side, k)?;
        let (q, status) = self.resolve(u, child_side, j, cap.min(remaining))?;
        if status == Status::Blocked {
            if cap < remaining {
                return Ok((0.0, Status::Blocked));
            }
            return self.local_step(v, side, k, cap);
        }
        self.scales.entry((v, k)).or_default().recursive_cost += q;
        if !self.failed {
            self.refresh_latch(k);
        }
        Ok((q, status))
    }
    fn separated(&self) -> Option<NodeId> {
        self.tree.children(0).iter().copied().find(|&a| {
            let competitor = self
                .tree
                .children(0)
                .iter()
                .filter(|&&u| u != a)
                .map(|&u| self.state[u].effective.upper)
                .fold(f64::NEG_INFINITY, f64::max);
            self.state[a].effective.lower >= competitor - self.config.epsilon
        })
    }
    /// Resource exits are inconclusive (no recommendation), never PAC claims.
    pub fn run(&mut self) -> Result<ResultReport, String> {
        if self.initialized {
            return Err("controller is single-use".into());
        }
        self.initialized = true;
        let start = Instant::now();
        let children = self.tree.children(0).to_vec();
        let stop;
        if children.len() as f64 > self.config.max_cost {
            stop = StopReason::CostBudget;
        } else {
            self.state[0].exposed = true;
            self.state[0].expanded = true;
            self.expose(&children)?;
            self.refresh();
            self.rho0 = children
                .iter()
                .map(|&v| self.state[v].effective.width())
                .fold(0.0, f64::max);
            loop {
                if self.failed {
                    stop = StopReason::ConfidenceFailure;
                    break;
                }
                if self.separated().is_some() {
                    stop = StopReason::Separated;
                    break;
                }
                if self.profile.rounds >= self.config.max_rounds {
                    stop = StopReason::RoundBudget;
                    break;
                }
                let (v, s, k) = match self.root_obligation() {
                    Ok(x) => x,
                    Err(_) => {
                        stop = StopReason::PrecisionLimit;
                        break;
                    }
                };
                let (q, status) = self.resolve(v, s, k, self.config.max_cost - self.cost)?;
                self.profile.rounds += 1;
                if status == Status::Blocked {
                    stop = StopReason::CostBudget;
                    break;
                }
                if q == 0.0 {
                    return Err("non-stopping resolver made zero progress".into());
                }
            }
        }
        self.profile.total_time = start.elapsed();
        let recommendation = match stop {
            StopReason::Separated => self.separated(),
            StopReason::ConfidenceFailure => children.first().copied(),
            _ => None,
        };
        Ok(ResultReport {
            recommendation,
            stop,
            oracle_contract_justified: self.contract.justified,
            total_cost: self.cost,
            root_intervals: children
                .iter()
                .filter(|&&v| self.state[v].exposed)
                .map(|&v| (v, self.state[v].effective))
                .collect(),
            profile: self.profile.clone(),
            trace: self.trace.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone)]
    struct Tree {
        children: Vec<Vec<usize>>,
    }
    impl FiniteTree for Tree {
        fn node_count(&self) -> usize {
            self.children.len()
        }
        fn children(&self, node: usize) -> &[usize] {
            &self.children[node]
        }
    }
    struct Fast(Vec<f64>);
    impl FastOracle for Fast {
        fn evaluate(&mut self, v: usize) -> Result<f64, String> {
            Ok(self.0[v])
        }
    }
    struct Slow(Vec<f64>);
    impl SlowOracle for Slow {
        fn sample(&mut self, v: usize) -> Result<f64, String> {
            Ok(self.0[v])
        }
    }
    fn binary(depth: usize) -> Tree {
        let n = (1 << (depth + 1)) - 1;
        Tree {
            children: (0..n)
                .map(|v| {
                    if 2 * v + 2 < n {
                        vec![2 * v + 1, 2 * v + 2]
                    } else {
                        vec![]
                    }
                })
                .collect(),
        }
    }
    fn exact_values(tree: &Tree, leaves: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; tree.node_count()];
        let mut i = 0;
        for v in (0..values.len()).rev() {
            if tree.children(v).is_empty() {
                values[v] = leaves[i % leaves.len()];
                i += 1;
            } else {
                let depth = ((v + 1) as f64).log2().floor() as usize;
                values[v] = tree.children(v).iter().map(|&u| values[u]).fold(
                    if depth.is_multiple_of(2) {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    },
                    if depth.is_multiple_of(2) {
                        f64::max
                    } else {
                        f64::min
                    },
                );
            }
        }
        values
    }
    fn controller(tree: Tree, values: Vec<f64>, sigma: f64) -> TwoFidelity<Tree, Fast, Slow> {
        let d = ((tree.node_count() + 1) as f64).log2() as usize - 1;
        let fast = (0..tree.node_count())
            .map(|v| {
                if tree.children(v).is_empty() {
                    values[v]
                } else {
                    0.0
                }
            })
            .collect();
        TwoFidelity::new(
            tree,
            Fast(fast),
            Slow(values),
            Config {
                slow_cost: 1.0,
                trace: true,
                ..Config::default()
            },
            OracleContract {
                envelope: (0..=d).map(|h| h as f64).collect(),
                sigma,
                slow_bias_bound: 0.0,
                provenance: "synthetic exact minimax plus zero noise".into(),
                justified: true,
            },
        )
        .unwrap()
    }
    #[test]
    fn exact_minimax_grid_and_interval_validity() {
        for d in 1..=5 {
            for seed in 0..12 {
                let tree = binary(d);
                let leaves: Vec<_> = (0..64)
                    .map(|i| ((i * 19 + seed * 17) % 97) as f64 / 97.0)
                    .collect();
                let values = exact_values(&tree, &leaves);
                let expected = values[1].max(values[2]);
                let mut c = controller(tree, values.clone(), 0.01);
                let report = c.run().unwrap();
                assert_eq!(report.stop, StopReason::Separated);
                assert_eq!(values[report.recommendation.unwrap()], expected);
                for (v, state) in c
                    .state
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter(|(_, s)| s.exposed)
                {
                    assert!(
                        state.effective.lower <= values[v] && values[v] <= state.effective.upper
                    );
                }
                for (&(v, k), flags) in &c.scales {
                    let x = c.state[v].effective;
                    let tolerance = c.rho(k) / 2.0 + 1e-12;
                    if flags.done[0] {
                        assert!(values[v] - x.lower <= tolerance);
                    }
                    if flags.done[1] {
                        assert!(x.upper - values[v] <= tolerance);
                    }
                }
                assert_eq!(report.profile.rounds as usize, report.trace.len());
                assert_eq!(
                    report.total_cost,
                    (report.profile.fast_queries + report.profile.slow_queries) as f64
                );
            }
        }
    }
    #[test]
    fn ties_and_epsilon_can_stop_without_unique_best() {
        let tree = binary(2);
        let values = vec![0.5; tree.node_count()];
        let mut c = controller(tree.clone(), values.clone(), 0.0);
        assert_eq!(c.run().unwrap().stop, StopReason::Separated);
        let mut c = controller(tree, values, 0.0);
        c.config.epsilon = 2.0;
        let report = c.run().unwrap();
        assert_eq!(report.profile.rounds, 0);
        assert_eq!(report.total_cost, 2.0);
    }
    #[test]
    fn zero_fast_envelope_is_immediate() {
        let tree = binary(1);
        let values = vec![0.7, 0.4, 0.7];
        let mut c = controller(tree, values, 0.02);
        let report = c.run().unwrap();
        assert_eq!(report.recommendation, Some(2));
        assert_eq!(report.profile.slow_queries, 0);
        assert_eq!(report.profile.rounds, 0);
    }
    #[test]
    fn running_intersection_and_expanded_samples_are_retained() {
        let tree = binary(3);
        let values = vec![0.2; tree.node_count()];
        let mut c = controller(tree, values, 0.1);
        c.state[0].exposed = true;
        c.state[0].expanded = true;
        c.expose(&[1, 2]).unwrap();
        c.refresh();
        c.rho0 = 4.0;
        c.local_step(1, Side::Lower, 0, 100.0).unwrap();
        let old_slow = c.state[1].slow;
        c.expand(1, Side::Lower, 0).unwrap();
        assert_eq!(c.slow_samples(1), 1);
        assert_eq!(c.state[1].slow, old_slow);
        c.local_step(1, Side::Upper, 0, 100.0).unwrap();
        assert_eq!(c.slow_samples(1), 2);
        assert!(c.state[1].slow.lower >= old_slow.lower && c.state[1].slow.upper <= old_slow.upper);
    }
    fn wide_tree(depth: usize, width: usize) -> (Tree, Vec<usize>) {
        let mut children = vec![vec![]];
        let mut depths = vec![0];
        let mut i = 0;
        while i < children.len() {
            if depths[i] < depth {
                for _ in 0..if i == 0 { 2 } else { width } {
                    let u = children.len();
                    children.push(vec![]);
                    depths.push(depths[i] + 1);
                    children[i].push(u);
                }
            }
            i += 1;
        }
        (Tree { children }, depths)
    }
    #[test]
    fn own_budget_falls_back_after_expansion() {
        let (tree, depths) = wide_tree(3, 8);
        let values: Vec<_> = depths
            .iter()
            .enumerate()
            .map(|(v, _)| {
                if v == 0 {
                    0.6
                } else if v == 1 || (v > 2 && v % 2 == 1) {
                    0.4
                } else {
                    0.6
                }
            })
            .collect();
        // Equal descendants per root subtree, computed independently by membership.
        let mut true_values = vec![0.0; tree.node_count()];
        fn assign(t: &Tree, vals: &mut [f64], v: usize, x: f64) {
            vals[v] = x;
            for &u in t.children(v) {
                assign(t, vals, u, x);
            }
        }
        assign(&tree, &mut true_values, 1, values[1]);
        assign(&tree, &mut true_values, 2, values[2]);
        true_values[0] = 0.6;
        let fast = depths
            .iter()
            .enumerate()
            .map(|(v, &d)| if d == 3 { true_values[v] } else { 0.0 })
            .collect();
        let mut c = TwoFidelity::new(
            tree,
            Fast(fast),
            Slow(true_values),
            Config {
                slow_cost: 1.0,
                trace: true,
                ..Config::default()
            },
            OracleContract {
                envelope: vec![0.0, 1.0, 2.0, 3.0],
                sigma: 0.0,
                slow_bias_bound: 0.0,
                provenance: "synthetic".into(),
                justified: true,
            },
        )
        .unwrap();
        // Resolve the same fine-scale obligation repeatedly, so expansion and
        // descendant work share its recursive counter (not a later scale).
        c.state[0].exposed = true;
        c.state[0].expanded = true;
        c.expose(&[1, 2]).unwrap();
        c.refresh();
        c.rho0 = 4.0;
        for _ in 0..20 {
            c.resolve(1, Side::Lower, 1, 100.0).unwrap();
            if c.slow_samples(1) > 0 {
                break;
            }
        }
        assert!(c.is_expanded(1));
        assert!(c.slow_samples(1) > 0);
        assert!(
            c.trace
                .iter()
                .any(|event| !event.expansion && event.node == 1)
        );
        for (&(v, k), s) in &c.scales {
            assert!(s.recursive_cost <= c.race_budget(v, k));
        }
    }
    #[test]
    fn inherited_cap_blocks_without_local_fallback() {
        let mut c = controller(binary(3), vec![0.2; 15], 0.0);
        c.state[0].exposed = true;
        c.state[0].expanded = true;
        c.expose(&[1, 2]).unwrap();
        c.refresh();
        c.rho0 = 4.0;
        let before = c.cost;
        let (q, status) = c.resolve(1, Side::Lower, 0, 0.5).unwrap();
        assert_eq!((q, status), (0.0, Status::Blocked));
        assert_eq!(c.cost, before);
        assert_eq!(c.slow_samples(1), 0);
    }
    #[test]
    fn resource_exits_do_not_claim_recommendations() {
        for budget in [0.0, 2.0, 2.5] {
            let mut c = controller(binary(3), vec![0.2; 15], 0.01);
            c.config.max_cost = budget;
            let r = c.run().unwrap();
            assert_eq!(r.stop, StopReason::CostBudget);
            assert!(r.recommendation.is_none());
            assert!(r.total_cost <= budget);
        }
        let mut c = controller(binary(3), vec![0.2; 15], 0.01);
        c.config.max_rounds = 0;
        assert_eq!(c.run().unwrap().stop, StopReason::RoundBudget);
        let mut c = controller(binary(3), vec![0.2; 15], 0.01);
        c.config.max_scale = 0;
        let r = c.run().unwrap();
        assert_eq!(r.stop, StopReason::PrecisionLimit);
        assert!(r.recommendation.is_none());
    }
    #[test]
    fn empty_intersection_returns_fixed_default_and_marks_failure() {
        let (tree, _) = wide_tree(2, 8);
        let n = tree.node_count();
        let mut c = TwoFidelity::new(
            tree,
            Fast(vec![0.0; n]),
            Slow(vec![5.0; n]),
            Config {
                slow_cost: 1.0,
                ..Config::default()
            },
            OracleContract {
                envelope: vec![0.0, 1.0, 2.0],
                sigma: 0.0,
                slow_bias_bound: 0.0,
                provenance: "deliberately false synthetic premise".into(),
                justified: false,
            },
        )
        .unwrap();
        let r = c.run().unwrap();
        assert_eq!(r.stop, StopReason::ConfidenceFailure);
        assert_eq!(r.recommendation, Some(1));
        assert!(!r.oracle_contract_justified);
    }
    #[test]
    fn confidence_radius_is_monotone_and_sample_inverse_minimal() {
        let c = controller(binary(3), vec![0.2; 15], 0.1);
        for n in 1..1000 {
            assert!(c.radius(n + 1) <= c.radius(n));
        }
        for target in [0.001, 0.01, 0.1, 1.0] {
            let n = c.required_samples(target).unwrap();
            assert!(c.radius(n) <= target);
            assert!(n == 1 || c.radius(n - 1) > target);
        }
    }
    #[test]
    fn malformed_topology_envelopes_and_nan_oracles_reject() {
        let bad = Tree {
            children: vec![vec![1, 2], vec![2], vec![]],
        };
        assert!(
            TwoFidelity::new(
                bad,
                Fast(vec![]),
                Slow(vec![]),
                Config::default(),
                OracleContract {
                    envelope: vec![0.0, 1.0],
                    sigma: 0.1,
                    slow_bias_bound: 0.0,
                    provenance: "x".into(),
                    justified: false
                }
            )
            .is_err()
        );
        let mut c = controller(binary(2), vec![0.2; 7], 0.0);
        c.fast.0[1] = f64::NAN;
        assert!(c.run().is_err());
    }
    #[test]
    fn oracle_batch_cardinality_and_errors_reject() {
        struct WrongBatch;
        impl FastOracle for WrongBatch {
            fn evaluate(&mut self, _: usize) -> Result<f64, String> {
                Err("not called".into())
            }
            fn evaluate_batch(&mut self, _: &[usize]) -> Result<Vec<f64>, String> {
                Ok(vec![])
            }
        }
        let contract = OracleContract {
            envelope: vec![0.0, 1.0],
            sigma: 0.1,
            slow_bias_bound: 0.0,
            provenance: "x".into(),
            justified: false,
        };
        let mut c = TwoFidelity::new(
            binary(1),
            WrongBatch,
            Slow(vec![0.0; 3]),
            Config::default(),
            contract,
        )
        .unwrap();
        assert!(c.run().unwrap_err().contains("batch size"));
    }
    #[test]
    fn known_slow_bias_inflation_preserves_true_value() {
        let tree = binary(3);
        let values = exact_values(&tree, &[0.2, 0.6, 0.3, 0.5]);
        let mut c = controller(tree, values.clone(), 0.0);
        c.contract.slow_bias_bound = 0.15;
        c.slow.0 = values.iter().map(|x| x + 0.1).collect();
        let r = c.run().unwrap();
        assert_eq!(r.stop, StopReason::Separated);
        assert_eq!(values[r.recommendation.unwrap()], values[1].max(values[2]));
        for (v, state) in c.state.iter().enumerate().filter(|(_, s)| s.exposed) {
            assert!(state.effective.lower <= values[v] && values[v] <= state.effective.upper);
        }
    }
}
