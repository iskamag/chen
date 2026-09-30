//! A common scheduler over bounded host/device tree arenas.
//! All node state, rules, evaluation and minimax operations belong to the arena.
use crate::{
    Move, Position, SearchLimits, SearchResult,
    backend::{BatchTelemetry, SearchBackend},
    eval::{Evaluator, Material, Tapered},
    rules::RuleState,
    search::{MATE, MAX_EVAL},
};
use std::{
    fmt,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeSpan {
    pub start: u64,
    pub count: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeStats {
    pub arena_positions: u64,
    pub classified_positions: u64,
    pub evaluated_positions: u64,
    pub next_start: u64,
    pub next_count: u64,
    pub kernel_ms: f64,
}
impl TreeStats {
    pub fn next_span(&self) -> TreeSpan {
        TreeSpan {
            start: self.next_start,
            count: self.next_count,
        }
    }
}
#[derive(Debug)]
pub enum TreeError {
    Capacity(String),
    Backend(String),
}
impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity(s) => write!(f, "arena capacity: {s}"),
            Self::Backend(s) => write!(f, "arena backend: {s}"),
        }
    }
}
impl std::error::Error for TreeError {}
impl From<String> for TreeError {
    fn from(s: String) -> Self {
        Self::Backend(s)
    }
}
pub trait ResidentTree: Send {
    fn name(&self) -> &str;
    fn reset(&mut self, root: RuleState, history: &[u64]) -> Result<(), TreeError>;
    fn classify(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError>;
    fn expand(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError>;
    fn evaluate(&mut self, span: TreeSpan, evaluator: u32) -> Result<TreeStats, TreeError>;
    fn reduce(&mut self, span: TreeSpan) -> Result<TreeStats, TreeError>;
    fn finish(&mut self) -> Result<(i32, Vec<Move>, TreeStats), TreeError>;
}
pub struct ResidentSearch {
    tree: Box<dyn ResidentTree>,
    evaluator: u32,
    stop: Arc<AtomicBool>,
    telemetry: BatchTelemetry,
}
impl ResidentSearch {
    pub fn new(tree: Box<dyn ResidentTree>, evaluator: &str) -> Result<Self, String> {
        let evaluator = match evaluator {
            "material" => 0,
            "tapered" => 1,
            _ => return Err("resident baseline evaluator must be material or tapered".into()),
        };
        Ok(Self {
            tree,
            evaluator,
            stop: Arc::new(AtomicBool::new(false)),
            telemetry: BatchTelemetry::default(),
        })
    }
    fn interrupted(&self, start: Instant, limits: &SearchLimits) -> bool {
        self.stop.load(Ordering::Relaxed) || limits.time.is_some_and(|cap| start.elapsed() >= cap)
    }
    fn account(
        &mut self,
        stats: TreeStats,
        nodes: &mut u64,
        prior: u64,
        telemetry_prior: BatchTelemetry,
    ) {
        *nodes = prior + stats.classified_positions;
        self.telemetry = BatchTelemetry {
            positions: telemetry_prior.positions + stats.evaluated_positions,
            kernel_ms: telemetry_prior.kernel_ms + stats.kernel_ms,
            ..telemetry_prior
        };
    }
    fn admit(
        &mut self,
        span: TreeSpan,
        limits: &SearchLimits,
        nodes: &mut u64,
        stats: &mut TreeStats,
        prior: u64,
        telemetry_prior: BatchTelemetry,
    ) -> Result<bool, TreeError> {
        if let Some(cap) = limits.nodes {
            let remaining = cap.saturating_sub(*nodes);
            if span.count > remaining {
                if remaining > 0 {
                    *stats = self.tree.classify(
                        TreeSpan {
                            start: span.start,
                            count: remaining,
                        },
                        true,
                    )?;
                    self.account(*stats, nodes, prior, telemetry_prior);
                }
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn iteration(
        &mut self,
        root: RuleState,
        depth: u8,
        history: &[u64],
        limits: &SearchLimits,
        start: Instant,
        nodes: &mut u64,
    ) -> Result<Option<(i32, Vec<Move>)>, TreeError> {
        if self.interrupted(start, limits) {
            return Ok(None);
        }
        self.tree.reset(root, history)?;
        let prior = *nodes;
        let telemetry_prior = self.telemetry;
        let mut stats = TreeStats::default();
        let mut levels = vec![TreeSpan { start: 0, count: 1 }];
        for _ in 0..depth {
            let span = *levels.last().unwrap();
            if self.interrupted(start, limits)
                || !self.admit(span, limits, nodes, &mut stats, prior, telemetry_prior)?
            {
                return Ok(None);
            }
            match self.tree.expand(span, true) {
                Ok(next) => stats = next,
                Err(TreeError::Capacity(_)) => {
                    // Overflow admits no children; classify the already submitted parents.
                    stats = self.tree.classify(span, true)?;
                    self.account(stats, nodes, prior, telemetry_prior);
                    return Ok(None);
                }
                Err(e) => return Err(e),
            }
            self.account(stats, nodes, prior, telemetry_prior);
            if stats.next_count == 0 {
                break;
            }
            levels.push(stats.next_span());
        }
        let leaf = *levels.last().unwrap();
        // A terminal frontier was already classified by expand; a depth-limited
        // frontier has not. Classified counts avoid charging terminal work twice.
        let unclassified = stats
            .arena_positions
            .saturating_sub(stats.classified_positions);
        if self.interrupted(start, limits) {
            return Ok(None);
        }
        if unclassified > 0
            && !self.admit(leaf, limits, nodes, &mut stats, prior, telemetry_prior)?
        {
            return Ok(None);
        }
        stats = self.tree.evaluate(leaf, self.evaluator)?;
        self.account(stats, nodes, prior, telemetry_prior);
        for &span in levels[..levels.len() - 1].iter().rev() {
            if self.interrupted(start, limits) {
                return Ok(None);
            }
            stats = self.tree.reduce(span)?;
            self.account(stats, nodes, prior, telemetry_prior);
        }
        if self.interrupted(start, limits) {
            return Ok(None);
        }
        let (score, pv, stats) = self.tree.finish()?;
        self.account(stats, nodes, prior, telemetry_prior);
        Ok(Some((score, pv)))
    }
}
impl SearchBackend for ResidentSearch {
    fn name(&self) -> &str {
        self.tree.name()
    }
    fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    fn clear(&mut self) {
        self.telemetry = BatchTelemetry::default();
    }
    fn batch_telemetry(&self) -> BatchTelemetry {
        self.telemetry
    }
    fn search(
        &mut self,
        p: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        let start = Instant::now();
        self.telemetry = BatchTelemetry::default();
        // Root validation and an emergency legal fallback are boundary operations.
        // Search iterations execute rules entirely within the chosen arena.
        let moves = p.legal_moves();
        let mut result = SearchResult {
            best_move: moves.first().copied(),
            score: if moves.is_empty() && p.is_in_check(p.side_to_move()) {
                -MATE
            } else {
                0
            },
            depth: 0,
            nodes: 0,
            qnodes: 0,
            elapsed: Duration::ZERO,
            pv: moves.first().copied().into_iter().collect(),
            stopped: false,
            hashfull: 0,
        };
        if moves.is_empty() {
            result.elapsed = start.elapsed();
            return Ok(result);
        }
        let mut nodes = 0;
        for depth in 1..=limits.depth.clamp(1, 64) {
            match self
                .iteration(
                    RuleState::from(&*p),
                    depth,
                    history,
                    &limits,
                    start,
                    &mut nodes,
                )
                .map_err(|e| e.to_string())?
            {
                Some((score, pv)) => {
                    result.score = score;
                    result.depth = depth;
                    if let Some(&mv) = pv.first() {
                        result.best_move = Some(mv);
                        result.pv = pv;
                    }
                    result.nodes = nodes;
                    result.elapsed = start.elapsed();
                    report(&result);
                    if score.abs() >= MATE - 128 {
                        break;
                    }
                }
                None => {
                    result.stopped = true;
                    break;
                }
            }
        }
        result.nodes = nodes;
        result.elapsed = start.elapsed();
        Ok(result)
    }
}
struct HostNode {
    p: Position,
    parent: Option<usize>,
    mv: Option<Move>,
    children: Range<usize>,
    moves: Option<Vec<Move>>,
    score: Option<i32>,
    best: Option<usize>,
    ply: usize,
    evaluated: bool,
    terminal: bool,
}
pub struct HostTree {
    nodes: Vec<HostNode>,
    history: Vec<u64>,
    capacity: usize,
    stats: TreeStats,
    mode: Option<bool>,
    evaluator: Option<u32>,
}
impl HostTree {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("tree capacity must be positive".into());
        }
        Ok(Self {
            nodes: Vec::new(),
            history: Vec::new(),
            capacity,
            stats: TreeStats::default(),
            mode: None,
            evaluator: None,
        })
    }
    fn range(&self, span: TreeSpan) -> Result<Range<usize>, TreeError> {
        if self.nodes.is_empty() {
            return Err(TreeError::Backend("arena not reset".into()));
        }
        let end = span
            .start
            .checked_add(span.count)
            .ok_or_else(|| TreeError::Backend("span overflow".into()))?;
        if end > self.nodes.len() as u64 {
            return Err(TreeError::Backend("span outside arena".into()));
        }
        Ok(span.start as usize..end as usize)
    }
    fn repeated(&self, mut index: usize) -> bool {
        let key = self.nodes[index].p.hash();
        let mut remain = self.nodes[index].p.halfmove_clock() as usize + 1;
        let mut count = 0;
        loop {
            if remain == 0 {
                break;
            }
            count += usize::from(self.nodes[index].p.hash() == key);
            remain -= 1;
            if let Some(parent) = self.nodes[index].parent {
                index = parent;
            } else {
                break;
            }
        }
        count += self
            .history
            .iter()
            .rev()
            .take(remain)
            .filter(|&&h| h == key)
            .count();
        count >= 3
    }
}
impl ResidentTree for HostTree {
    fn name(&self) -> &str {
        "cpu-resident"
    }
    fn reset(&mut self, root: RuleState, history: &[u64]) -> Result<(), TreeError> {
        let p = root.to_position()?;
        if history.len() > u16::MAX as usize {
            return Err(TreeError::Backend("history exceeds ABI capacity".into()));
        }
        self.nodes.clear();
        self.nodes.push(HostNode {
            p,
            parent: None,
            mv: None,
            children: 0..0,
            moves: None,
            score: None,
            best: None,
            ply: 0,
            evaluated: false,
            terminal: false,
        });
        self.history = history.to_vec();
        self.mode = None;
        self.evaluator = None;
        self.stats = TreeStats {
            arena_positions: 1,
            ..TreeStats::default()
        };
        Ok(())
    }
    fn classify(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError> {
        let range = self.range(span)?;
        if self.mode.is_some_and(|mode| mode != adjudicate) {
            return Err(TreeError::Backend(
                "classification mode changed; reset arena".into(),
            ));
        }
        if range.clone().any(|i| !self.nodes[i].children.is_empty()) {
            return Err(TreeError::Backend("cannot classify expanded nodes".into()));
        }
        self.mode = Some(adjudicate);
        for index in range {
            if self.nodes[index].moves.is_some() {
                continue;
            }
            let p = self.nodes[index].p;
            let moves = p.legal_moves();
            self.nodes[index].score = if moves.is_empty() {
                Some(if p.is_in_check(p.side_to_move()) {
                    -MATE + self.nodes[index].ply as i32
                } else {
                    0
                })
            } else if adjudicate
                && (p.halfmove_clock() >= 100 || p.insufficient_material() || self.repeated(index))
            {
                Some(0)
            } else {
                None
            };
            self.nodes[index].moves = Some(moves);
            self.nodes[index].terminal = self.nodes[index].score.is_some();
            self.stats.classified_positions += 1;
        }
        Ok(self.stats)
    }
    fn expand(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError> {
        let range = self.range(span)?;
        if range
            .clone()
            .any(|i| !self.nodes[i].children.is_empty() || self.nodes[i].evaluated)
        {
            return Err(TreeError::Backend(
                "node already expanded or statically evaluated".into(),
            ));
        }
        self.classify(span, adjudicate)?;
        let extra: usize = range
            .clone()
            .map(|i| {
                if adjudicate && self.nodes[i].score.is_some() {
                    0
                } else {
                    self.nodes[i].moves.as_ref().unwrap().len()
                }
            })
            .sum();
        if extra > self.capacity.saturating_sub(self.nodes.len()) {
            return Err(TreeError::Capacity("children exceed tree capacity".into()));
        }
        let start = self.nodes.len();
        for index in range {
            if adjudicate && self.nodes[index].score.is_some() {
                continue;
            }
            let moves = self.nodes[index].moves.as_ref().unwrap().clone();
            let from = self.nodes.len();
            for mv in moves {
                let mut p = self.nodes[index].p;
                p.make_move(mv);
                self.nodes.push(HostNode {
                    p,
                    parent: Some(index),
                    mv: Some(mv),
                    children: 0..0,
                    moves: None,
                    score: None,
                    best: None,
                    ply: self.nodes[index].ply + 1,
                    evaluated: false,
                    terminal: false,
                });
            }
            self.nodes[index].children = from..self.nodes.len();
        }
        self.stats.arena_positions = self.nodes.len() as u64;
        self.stats.next_start = start as u64;
        self.stats.next_count = (self.nodes.len() - start) as u64;
        Ok(self.stats)
    }
    fn evaluate(&mut self, span: TreeSpan, evaluator: u32) -> Result<TreeStats, TreeError> {
        if evaluator > 1 {
            return Err(TreeError::Backend("unknown evaluator".into()));
        }
        if self.evaluator.is_some_and(|mode| mode != evaluator) {
            return Err(TreeError::Backend("evaluator changed; reset arena".into()));
        }
        let range = self.range(span)?;
        if range.clone().any(|i| !self.nodes[i].children.is_empty()) {
            return Err(TreeError::Backend("cannot evaluate expanded nodes".into()));
        }
        self.classify(span, true)?;
        self.evaluator = Some(evaluator);
        for index in range {
            if self.nodes[index].score.is_none() {
                let p = self.nodes[index].p;
                let score = if evaluator == 0 {
                    Material.evaluate(&p)
                } else {
                    Tapered.evaluate(&p)
                };
                self.nodes[index].score = Some(score.clamp(-MAX_EVAL, MAX_EVAL));
                self.nodes[index].evaluated = true;
                self.stats.evaluated_positions += 1;
            }
        }
        Ok(self.stats)
    }
    fn reduce(&mut self, span: TreeSpan) -> Result<TreeStats, TreeError> {
        let range = self.range(span)?;
        if let Some(first) = range.clone().next()
            && range
                .clone()
                .any(|i| self.nodes[i].ply != self.nodes[first].ply)
        {
            return Err(TreeError::Backend(
                "reduction spans multiple tree levels".into(),
            ));
        }
        for index in range {
            let range = self.nodes[index].children.clone();
            if range.is_empty() {
                if !self.nodes[index].terminal {
                    return Err(TreeError::Backend(
                        "unexpanded nonterminal in reduction".into(),
                    ));
                }
                continue;
            }
            let mut best = i32::MIN;
            let mut child_best = None;
            for child in range {
                let score = -self.nodes[child]
                    .score
                    .ok_or_else(|| TreeError::Backend("child score missing".into()))?;
                if score > best {
                    best = score;
                    child_best = Some(child);
                }
            }
            self.nodes[index].score = Some(best);
            self.nodes[index].best = child_best;
        }
        Ok(self.stats)
    }
    fn finish(&mut self) -> Result<(i32, Vec<Move>, TreeStats), TreeError> {
        let root = self
            .nodes
            .first()
            .ok_or_else(|| TreeError::Backend("arena not reset".into()))?;
        let score = root
            .score
            .ok_or_else(|| TreeError::Backend("root not scored".into()))?;
        let mut pv = Vec::new();
        let mut index = 0;
        while let Some(child) = self.nodes[index].best {
            pv.push(self.nodes[child].mv.unwrap());
            index = child;
        }
        Ok((score, pv, self.stats))
    }
}
