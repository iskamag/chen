//! Bounded best-open minimax over complete legal sibling sets.
//!
//! Selection and mutation belong to the arena, so the scheduler transfers only
//! the root/history, budgets and a final PV. CPU and device arenas share this
//! contract; neither mixed-depth score nor deepest frontier is an alpha-beta
//! horizon claim. Terminal positions are closed and still participate in backup.
use crate::{
    Move, Position, SearchLimits, SearchResult,
    backend::{BatchTelemetry, NeuralExecutionStats, SearchBackend},
    eval::{Evaluator, Material, Tapered},
    resident::TreeError,
    rules::RuleState,
    search::{MATE, MAX_EVAL},
};
use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

pub const SELECTIVE_ABI: u32 = 1;
pub const PUBLISHED: u32 = 1;
pub const TERMINAL: u32 = 2;
pub const HORIZON_COMPLETE: u32 = 4;
pub const MATE_PROOF: u32 = 8;
pub const RUNNING: u32 = 0;
pub const CLOSED: u32 = 1;
pub const NODE_LIMIT: u32 = 2;
pub const CAPACITY_LIMIT: u32 = 3;
pub const CANDIDATE_LIMIT: u32 = 4;
pub const FALLBACK: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SelectiveConfig {
    pub abi_version: u32,
    pub struct_size: u32,
    pub max_ply: u32,
    pub policy: u32,
    pub batch_leaves: u32,
    pub candidate_limit: u32,
    pub reserved: u64,
}
impl Default for SelectiveConfig {
    fn default() -> Self {
        Self {
            abi_version: SELECTIVE_ABI,
            struct_size: std::mem::size_of::<Self>() as u32,
            max_ply: 4,
            policy: 0,
            batch_leaves: 32,
            candidate_limit: 1024,
            reserved: 0,
        }
    }
}
impl SelectiveConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.abi_version != SELECTIVE_ABI
            || self.struct_size as usize != std::mem::size_of::<Self>()
            || !(1..=64).contains(&self.max_ply)
            || self.policy != 0
            || !(1..=256).contains(&self.batch_leaves)
            || !(self.batch_leaves..=1024).contains(&self.candidate_limit)
            || self.reserved != 0
        {
            return Err("invalid selective ABI/configuration".into());
        }
        Ok(())
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelectiveStats {
    pub abi_version: u32,
    pub struct_size: u32,
    pub arena_positions: u64,
    pub classified_positions: u64,
    pub evaluated_positions: u64,
    pub completed_steps: u64,
    pub expanded_leaves: u64,
    pub open_frontier_leaves: u64,
    pub last_selected: u32,
    pub last_added: u32,
    pub kernel_ms: f64,
    pub minimum_frontier_depth: u32,
    pub maximum_frontier_depth: u32,
    pub pv_depth: u32,
    pub root_flags: u32,
    pub stop_reason: u32,
    pub reserved: u32,
}
impl SelectiveStats {
    pub fn empty() -> Self {
        Self {
            abi_version: SELECTIVE_ABI,
            struct_size: std::mem::size_of::<Self>() as u32,
            ..Self::default()
        }
    }
    pub fn published(&self) -> bool {
        self.root_flags & PUBLISHED != 0
    }
    pub fn closed(&self) -> bool {
        self.root_flags & HORIZON_COMPLETE != 0
    }
}
/// Each successful advance is an atomic publication boundary. A numerical or
/// backend failure invalidates the arena until reset; finish must reject it.
pub trait SelectiveTree: Send {
    fn name(&self) -> &str;
    fn evaluator_artifact_sha256(&self) -> Option<&str> {
        None
    }
    fn neural_execution_stats(&self) -> Result<Option<NeuralExecutionStats>, String> {
        Ok(None)
    }
    fn reset(
        &mut self,
        root: RuleState,
        history: &[u64],
        config: SelectiveConfig,
    ) -> Result<SelectiveStats, TreeError>;
    fn advance(
        &mut self,
        absolute_node_limit: u64,
        evaluator: u32,
    ) -> Result<SelectiveStats, TreeError>;
    fn finish(&mut self) -> Result<(i32, Vec<Move>, SelectiveStats), TreeError>;
}

pub struct SelectiveSearch {
    tree: Box<dyn SelectiveTree>,
    evaluator: u32,
    config: SelectiveConfig,
    stop: Arc<AtomicBool>,
    telemetry: BatchTelemetry,
}
impl SelectiveSearch {
    pub fn new(
        tree: Box<dyn SelectiveTree>,
        evaluator: &str,
        config: SelectiveConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let evaluator = match evaluator {
            "material" => 0,
            "tapered" => 1,
            "neural" => 2,
            _ => return Err("unknown selective evaluator".into()),
        };
        Ok(Self {
            tree,
            evaluator,
            config,
            stop: Arc::new(AtomicBool::new(false)),
            telemetry: BatchTelemetry::default(),
        })
    }
}
impl SearchBackend for SelectiveSearch {
    fn name(&self) -> &str {
        self.tree.name()
    }
    fn evaluator_artifact_sha256(&self) -> Option<&str> {
        self.tree.evaluator_artifact_sha256()
    }
    fn neural_execution_stats(&self) -> Result<Option<NeuralExecutionStats>, String> {
        self.tree.neural_execution_stats()
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
        let legal = p.legal_moves();
        let mut result = SearchResult {
            best_move: legal.first().copied(),
            score: if legal.is_empty() && p.is_in_check(p.side_to_move()) {
                -MATE
            } else {
                0
            },
            depth: 0,
            nodes: 0,
            qnodes: 0,
            elapsed: Duration::ZERO,
            pv: legal.first().copied().into_iter().collect(),
            stopped: false,
            hashfull: 0,
            selective: None,
        };
        let config = SelectiveConfig {
            max_ply: u32::from(limits.depth.clamp(1, 64)),
            ..self.config
        };
        let mut stats = self
            .tree
            .reset(RuleState::from(&*p), history, config)
            .map_err(|e| e.to_string())?;
        loop {
            if self.stop.load(Ordering::Relaxed)
                || limits.time.is_some_and(|t| start.elapsed() >= t)
            {
                result.stopped = true;
                break;
            }
            stats = self
                .tree
                .advance(limits.nodes.unwrap_or(u64::MAX), self.evaluator)
                .map_err(|e| e.to_string())?;
            self.telemetry.positions = stats.evaluated_positions;
            self.telemetry.kernel_ms = stats.kernel_ms;
            if stats.published() {
                let (score, pv, finished) = self.tree.finish().map_err(|e| e.to_string())?;
                stats = finished;
                result.score = score;
                result.best_move = pv.first().copied();
                result.pv = pv;
                // A closed finite tree agrees with a full-width max-ply search.
                // Otherwise report only the conservative minimum retained ply.
                result.depth = if stats.root_flags & TERMINAL != 0 {
                    0
                } else if stats.closed() {
                    config.max_ply as u8
                } else {
                    stats.minimum_frontier_depth as u8
                };
                result.nodes = stats.classified_positions;
                result.elapsed = start.elapsed();
                result.selective = Some(stats);
                report(&result);
            }
            if stats.stop_reason != RUNNING || stats.root_flags & MATE_PROOF != 0 {
                result.stopped = !stats.closed() && stats.root_flags & MATE_PROOF == 0;
                break;
            }
        }
        result.nodes = stats.classified_positions;
        result.selective = Some(stats);
        self.telemetry.positions = stats.evaluated_positions;
        self.telemetry.kernel_ms = stats.kernel_ms;
        result.elapsed = start.elapsed();
        Ok(result)
    }
}

struct Node {
    position: Position,
    parent: Option<usize>,
    mv: Option<Move>,
    children: Range<usize>,
    moves: Vec<Move>,
    score: i32,
    best: Option<usize>,
    ply: u32,
    terminal: bool,
    open: u64,
}
/// Independent CPU implementation of the device scheduler, for exact replay.
pub struct HostSelectiveTree {
    nodes: Vec<Node>,
    history: Vec<u64>,
    capacity: usize,
    config: SelectiveConfig,
    stats: SelectiveStats,
    evaluator: Option<u32>,
    valid: bool,
}
impl HostSelectiveTree {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("selective tree capacity must be positive".into());
        }
        Ok(Self {
            nodes: Vec::new(),
            history: Vec::new(),
            capacity,
            config: SelectiveConfig::default(),
            stats: SelectiveStats::default(),
            evaluator: None,
            valid: false,
        })
    }
    fn repeated(&self, p: &Position, mut parent: Option<usize>) -> bool {
        let mut remain = p.halfmove_clock() as usize;
        let mut count = 1;
        while remain > 0 {
            let Some(index) = parent else { break };
            count += usize::from(self.nodes[index].position.hash() == p.hash());
            remain -= 1;
            parent = self.nodes[index].parent;
        }
        count += self
            .history
            .iter()
            .rev()
            .take(remain)
            .filter(|&&h| h == p.hash())
            .count();
        count >= 3
    }
    fn classify(
        &mut self,
        p: Position,
        parent: Option<usize>,
        mv: Option<Move>,
        ply: u32,
        evaluator: u32,
    ) -> Node {
        let moves = p.legal_moves();
        let terminal = moves.is_empty()
            || p.halfmove_clock() >= 100
            || p.insufficient_material()
            || self.repeated(&p, parent);
        let score = if moves.is_empty() && p.is_in_check(p.side_to_move()) {
            -MATE + ply as i32
        } else if terminal || parent.is_none() {
            0
        } else {
            self.stats.evaluated_positions += 1;
            match evaluator {
                0 => Material.evaluate(&p),
                _ => Tapered.evaluate(&p),
            }
            .clamp(-MAX_EVAL, MAX_EVAL)
        };
        self.stats.classified_positions += 1;
        Node {
            position: p,
            parent,
            mv,
            children: 0..0,
            moves,
            score,
            best: None,
            ply,
            terminal,
            open: u64::from(!terminal && ply < self.config.max_ply),
        }
    }
    fn selected(&self, reservations: &[u64]) -> Option<usize> {
        if self.nodes[0].open == reservations[0] {
            return None;
        }
        let mut index = 0;
        loop {
            if self.nodes[index].children.is_empty() {
                return Some(index);
            }
            let mut best = None;
            let mut score = i32::MIN;
            for child in self.nodes[index].children.clone() {
                if self.nodes[child].open > reservations[child] && -self.nodes[child].score > score
                {
                    score = -self.nodes[child].score;
                    best = Some(child);
                }
            }
            index = best.expect("open count guarantees an available child");
        }
    }
    fn backup(&mut self) {
        for index in (0..self.nodes.len()).rev() {
            if self.nodes[index].children.is_empty() {
                continue;
            }
            let mut best_score = i32::MIN;
            let mut best = None;
            let mut open = 0;
            for child in self.nodes[index].children.clone() {
                open += self.nodes[child].open;
                if -self.nodes[child].score > best_score {
                    best_score = -self.nodes[child].score;
                    best = Some(child);
                }
            }
            self.nodes[index].score = best_score;
            self.nodes[index].best = best;
            self.nodes[index].open = open;
        }
    }
    fn refresh(&mut self) {
        self.stats.arena_positions = self.nodes.len() as u64;
        self.stats.open_frontier_leaves = self.nodes[0].open;
        let mut minimum = u32::MAX;
        let mut maximum = 0;
        for node in &self.nodes {
            if node.children.is_empty() {
                minimum = minimum.min(node.ply);
                maximum = maximum.max(node.ply);
            }
        }
        self.stats.minimum_frontier_depth = minimum;
        self.stats.maximum_frontier_depth = maximum;
        let mut index = 0;
        let mut pv_depth = 0;
        while let Some(child) = self.nodes[index].best {
            pv_depth += 1;
            index = child;
        }
        self.stats.pv_depth = pv_depth;
        self.stats.root_flags = if self.nodes[0].terminal {
            PUBLISHED | TERMINAL
        } else if self.nodes[0].children.is_empty() {
            0
        } else {
            PUBLISHED
        };
        if self.nodes[0].open == 0 {
            self.stats.root_flags |= HORIZON_COMPLETE;
        }
        if self.stats.published() && self.nodes[0].score.abs() >= MATE - 64 {
            self.stats.root_flags |= MATE_PROOF;
        }
    }
}
impl SelectiveTree for HostSelectiveTree {
    fn name(&self) -> &str {
        "cpu-selective"
    }
    fn reset(
        &mut self,
        root: RuleState,
        history: &[u64],
        config: SelectiveConfig,
    ) -> Result<SelectiveStats, TreeError> {
        self.valid = false;
        config.validate()?;
        let position = root.to_position()?;
        if history.len() > u16::MAX as usize {
            return Err(TreeError::Backend("history exceeds ABI capacity".into()));
        }
        self.nodes.clear();
        self.history = history.to_vec();
        self.config = config;
        self.stats = SelectiveStats {
            arena_positions: 1,
            stop_reason: FALLBACK,
            ..SelectiveStats::empty()
        };
        self.evaluator = None;
        // Root classification is deferred so an absolute zero-node budget does
        // no uncharged rules/evaluation work in the arena.
        self.nodes.push(Node {
            position,
            parent: None,
            mv: None,
            children: 0..0,
            moves: Vec::new(),
            score: 0,
            best: None,
            ply: 0,
            terminal: false,
            open: 1,
        });
        self.valid = true;
        Ok(self.stats)
    }
    fn advance(
        &mut self,
        absolute_node_limit: u64,
        evaluator: u32,
    ) -> Result<SelectiveStats, TreeError> {
        if !self.valid {
            return Err(TreeError::Backend(
                "selective arena invalid; reset required".into(),
            ));
        }
        if evaluator > 1 || self.evaluator.is_some_and(|prior| prior != evaluator) {
            self.valid = false;
            return Err(TreeError::Backend(
                "unsupported/changed selective evaluator; reset required".into(),
            ));
        }
        if absolute_node_limit < self.stats.classified_positions {
            self.valid = false;
            return Err(TreeError::Backend(
                "node limit is below charged work; reset required".into(),
            ));
        }
        self.evaluator = Some(evaluator);
        self.stats.last_selected = 0;
        self.stats.last_added = 0;
        self.stats.stop_reason = RUNNING;
        if self.stats.classified_positions == 0 {
            if absolute_node_limit == 0 {
                self.stats.stop_reason = NODE_LIMIT;
                return Ok(self.stats);
            }
            self.nodes[0] = self.classify(self.nodes[0].position, None, None, 0, evaluator);
        }
        self.refresh();
        if self.nodes[0].open == 0 {
            self.stats.stop_reason = CLOSED;
            return Ok(self.stats);
        }
        if absolute_node_limit == self.stats.classified_positions {
            self.stats.stop_reason = NODE_LIMIT;
            return Ok(self.stats);
        }
        if self.nodes.len() == self.capacity {
            self.stats.stop_reason = CAPACITY_LIMIT;
            return Ok(self.stats);
        }
        let mut reservations = vec![0; self.nodes.len()];
        let mut selected = Vec::new();
        let mut added = 0usize;
        let mut scanned = 0u32;
        let mut blocked_nodes = false;
        let mut blocked_capacity = false;
        while scanned < self.config.candidate_limit
            && selected.len() < self.config.batch_leaves as usize
        {
            let Some(index) = self.selected(&reservations) else {
                break;
            };
            scanned += 1;
            let mut ancestor = Some(index);
            while let Some(i) = ancestor {
                reservations[i] += 1;
                ancestor = self.nodes[i].parent;
            }
            let count = self.nodes[index].moves.len();
            let fits_nodes = count as u64
                <= absolute_node_limit - self.stats.classified_positions - added as u64;
            let fits_capacity = count <= self.capacity - self.nodes.len() - added;
            blocked_nodes |= !fits_nodes;
            blocked_capacity |= !fits_capacity;
            if fits_nodes && fits_capacity {
                selected.push(index);
                added += count;
            }
        }
        if selected.is_empty() {
            self.stats.stop_reason = if scanned == self.config.candidate_limit
                && self.selected(&reservations).is_some()
            {
                CANDIDATE_LIMIT
            } else if blocked_nodes {
                NODE_LIMIT
            } else if blocked_capacity {
                CAPACITY_LIMIT
            } else {
                CANDIDATE_LIMIT
            };
            return Ok(self.stats);
        }
        self.stats.last_selected = selected.len() as u32;
        self.stats.last_added = added as u32;
        for index in selected {
            let first = self.nodes.len();
            let moves = self.nodes[index].moves.clone();
            for mv in moves {
                let mut position = self.nodes[index].position;
                position.make_move(mv);
                let child = self.classify(
                    position,
                    Some(index),
                    Some(mv),
                    self.nodes[index].ply + 1,
                    evaluator,
                );
                self.nodes.push(child);
            }
            self.nodes[index].children = first..self.nodes.len();
            self.stats.expanded_leaves += 1;
        }
        self.backup();
        self.stats.completed_steps += 1;
        self.refresh();
        if self.stats.closed() {
            self.stats.stop_reason = CLOSED;
        }
        Ok(self.stats)
    }
    fn finish(&mut self) -> Result<(i32, Vec<Move>, SelectiveStats), TreeError> {
        if !self.valid || !self.stats.published() {
            return Err(TreeError::Backend(
                "selective root has no valid complete publication".into(),
            ));
        }
        let mut pv = Vec::new();
        let mut index = 0;
        while let Some(child) = self.nodes[index].best {
            pv.push(self.nodes[child].mv.unwrap());
            index = child;
        }
        Ok((self.nodes[0].score, pv, self.stats))
    }
}
