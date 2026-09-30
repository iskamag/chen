//! Fail-soft negamax baseline with independently switchable search features.
use crate::{
    core::{Move, Piece, Position},
    eval::{Evaluator, VALUES},
    tt::{Bound, Entry, TranspositionTable},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
pub const MATE: i32 = 30_000;
pub const MAX_EVAL: i32 = 29_000;
const INF: i32 = 32_000;
const MAX_PLY: usize = 128;

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub tt: bool,
    pub quiescence: bool,
    pub move_ordering: bool,
    pub hash_mb: usize,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            tt: true,
            quiescence: true,
            move_ordering: true,
            hash_mb: 16,
        }
    }
}
#[derive(Clone, Debug)]
pub struct SearchLimits {
    pub depth: u8,
    pub nodes: Option<u64>,
    pub time: Option<Duration>,
}
impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            depth: 64,
            nodes: None,
            time: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub qnodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
    pub stopped: bool,
    pub hashfull: usize,
    /// Mixed-depth arena telemetry. Absent for uniform-depth backends.
    pub selective: Option<crate::selective::SelectiveStats>,
}
impl SearchResult {
    pub fn nps(&self) -> u64 {
        (self.nodes as f64 / self.elapsed.as_secs_f64().max(0.000001)) as u64
    }
    pub fn uci_info(&self) -> String {
        let score = if self.score.abs() >= MATE - MAX_PLY as i32 {
            let plies = MATE - self.score.abs();
            format!("mate {}", self.score.signum() * ((plies + 1) / 2))
        } else {
            format!("cp {}", self.score)
        };
        let selective = self.selective.map_or_else(String::new, |s| {
            format!(" seldepth {}", s.maximum_frontier_depth)
        });
        format!(
            "info depth {}{} score {} nodes {} nps {} hashfull {} time {} pv {}",
            self.depth,
            selective,
            score,
            self.nodes,
            self.nps(),
            self.hashfull,
            self.elapsed.as_millis(),
            self.pv
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}
pub struct Searcher {
    config: SearchConfig,
    evaluator: Box<dyn Evaluator>,
    table: TranspositionTable,
    stop: Arc<AtomicBool>,
    started: Instant,
    limits: SearchLimits,
    nodes: u64,
    qnodes: u64,
    aborted: bool,
    history: Vec<u64>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    quiet_history: [[i32; 64]; 64],
}
impl Searcher {
    pub fn new(evaluator: Box<dyn Evaluator>, config: SearchConfig) -> Self {
        let table = TranspositionTable::new(config.hash_mb);
        Self {
            config,
            evaluator,
            table,
            stop: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
            limits: SearchLimits::default(),
            nodes: 0,
            qnodes: 0,
            aborted: false,
            history: Vec::new(),
            killers: [[None; 2]; MAX_PLY],
            quiet_history: [[0; 64]; 64],
        }
    }
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }
    /// Policy changes invalidate scores and ordering state from earlier experiments.
    pub fn set_config(&mut self, config: SearchConfig) {
        if config.hash_mb != self.config.hash_mb {
            self.table = TranspositionTable::new(config.hash_mb);
        }
        self.config = config;
        self.clear();
    }
    pub fn set_evaluator(&mut self, evaluator: Box<dyn Evaluator>) {
        self.evaluator = evaluator;
        self.clear();
    }
    fn static_eval(&mut self, position: &Position) -> i32 {
        self.evaluator.evaluate(position).clamp(-MAX_EVAL, MAX_EVAL)
    }
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    pub fn clear(&mut self) {
        self.table.clear();
        self.quiet_history = [[0; 64]; 64];
        self.killers = [[None; 2]; MAX_PLY];
    }
    pub fn evaluator_name(&self) -> &'static str {
        self.evaluator.name()
    }
    /// History contains previous positions only; the current root is appended here.
    pub fn search(
        &mut self,
        p: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        mut report: impl FnMut(&SearchResult),
    ) -> SearchResult {
        self.started = Instant::now();
        self.limits = limits;
        self.nodes = 0;
        self.qnodes = 0;
        self.aborted = false;
        // The caller clears the stop flag before starting a job, avoiding a startup race.
        self.history = history.to_vec();
        self.history.push(p.hash());
        self.evaluator.reset(p);
        self.table.new_search();
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
            selective: None,
        };
        if !moves.is_empty() {
            for depth in 1..=self.limits.depth.clamp(1, 64) {
                let mut pv = Vec::new();
                let score = self.negamax(p, depth as i32, 0, -INF, INF, &mut pv);
                if self.aborted {
                    break;
                }
                result.score = score;
                result.depth = depth;
                if let Some(mv) = pv.first() {
                    result.best_move = Some(*mv);
                    result.pv = pv;
                }
                result.nodes = self.nodes;
                result.qnodes = self.qnodes;
                result.elapsed = self.started.elapsed();
                result.hashfull = self.table.hashfull();
                report(&result);
                if score.abs() >= MATE - MAX_PLY as i32 {
                    break;
                }
            }
        }
        result.nodes = self.nodes;
        result.qnodes = self.qnodes;
        result.elapsed = self.started.elapsed();
        result.stopped = self.aborted;
        result
    }
    fn visit(&mut self, q: bool) -> bool {
        if self.stop.load(Ordering::Relaxed)
            || self.limits.nodes.is_some_and(|n| self.nodes >= n)
            || (self.nodes & 255 == 0
                && self
                    .limits
                    .time
                    .is_some_and(|t| self.started.elapsed() >= t))
        {
            self.aborted = true;
            return false;
        }
        self.nodes += 1;
        if q {
            self.qnodes += 1;
        }
        true
    }
    fn draw(&self, p: &Position) -> bool {
        p.halfmove_clock() >= 100
            || p.insufficient_material()
            || self
                .history
                .iter()
                .rev()
                .take(p.halfmove_clock() as usize + 1)
                .filter(|&&h| h == p.hash())
                .count()
                >= 3
    }
    // Draw context is mixed into TT identity; entries cannot cross different game histories.
    fn tt_key(&self, p: &Position) -> u64 {
        self.history
            .iter()
            .rev()
            .take(p.halfmove_clock() as usize + 1)
            .fold(
                p.hash() ^ (p.halfmove_clock() as u64).wrapping_mul(0x9e3779b97f4a7c15),
                |acc, h| acc.rotate_left(7) ^ h.wrapping_mul(0x517cc1b727220a95),
            )
    }
    fn capture(p: &Position, mv: Move) -> bool {
        p.piece_at(mv.to).is_some()
            || (p
                .piece_at(mv.from)
                .is_some_and(|(_, piece)| piece == Piece::Pawn)
                && mv.from % 8 != mv.to % 8)
    }
    fn ordered(&self, p: &Position, moves: &mut [Move], hash_move: Option<Move>, ply: usize) {
        if !self.config.move_ordering {
            return;
        }
        moves.sort_by_cached_key(|&mv| {
            let score = if Some(mv) == hash_move {
                2_000_000
            } else if Self::capture(p, mv) {
                let victim = p
                    .piece_at(mv.to)
                    .map_or(100, |(_, piece)| VALUES[piece.index()]);
                let attacker = p
                    .piece_at(mv.from)
                    .map_or(0, |(_, piece)| VALUES[piece.index()]);
                1_000_000 + victim * 10 - attacker
                    + mv.promotion.map_or(0, |piece| VALUES[piece.index()])
            } else if mv.promotion.is_some() {
                900_000
            } else if self.killers[ply][0] == Some(mv) {
                800_000
            } else if self.killers[ply][1] == Some(mv) {
                700_000
            } else {
                self.quiet_history[mv.from as usize][mv.to as usize]
            };
            -score
        });
    }
    fn negamax(
        &mut self,
        p: &mut Position,
        depth: i32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        pv: &mut Vec<Move>,
    ) -> i32 {
        if depth <= 0 && self.config.quiescence {
            return self.quiesce(p, ply, alpha, beta, pv);
        }
        if !self.visit(false) {
            return 0;
        }
        let mut moves = p.legal_moves();
        if moves.is_empty() {
            return if p.is_in_check(p.side_to_move()) {
                -MATE + ply as i32
            } else {
                0
            };
        }
        if self.draw(p) {
            return 0;
        }
        if ply >= MAX_PLY - 1 || depth <= 0 {
            return self.static_eval(p);
        }
        let key = self.tt_key(p);
        let entry = if self.config.tt {
            self.table.get(key)
        } else {
            None
        };
        if ply > 0
            && let Some(e) = entry
            && e.depth as i32 >= depth
        {
            let score = from_tt(e.score, ply);
            if e.bound == Bound::Exact
                || (e.bound == Bound::Lower && score >= beta)
                || (e.bound == Bound::Upper && score <= alpha)
            {
                if let Some(mv) = e.best
                    && moves.contains(&mv)
                {
                    pv.push(mv);
                }
                return score;
            }
        }
        self.ordered(p, &mut moves, entry.and_then(|e| e.best), ply);
        let original_alpha = alpha;
        let mut best = -INF;
        let mut best_move = None;
        for mv in moves {
            let quiet = !Self::capture(p, mv) && mv.promotion.is_none();
            let undo = p.make_move(mv);
            self.evaluator.after_make(p, mv);
            self.history.push(p.hash());
            let mut child = Vec::new();
            let score = -self.negamax(p, depth - 1, ply + 1, -beta, -alpha, &mut child);
            self.history.pop();
            p.unmake_move(mv, undo);
            self.evaluator.after_unmake(p, mv);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                best_move = Some(mv);
                pv.clear();
                pv.push(mv);
                pv.extend(child);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                if quiet {
                    if self.killers[ply][0] != Some(mv) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(mv);
                    }
                    let h = &mut self.quiet_history[mv.from as usize][mv.to as usize];
                    *h = (*h + depth * depth).min(600_000);
                }
                break;
            }
        }
        if self.config.tt {
            self.table.store(Entry {
                key,
                depth: depth as u8,
                score: to_tt(best, ply),
                bound: if best <= original_alpha {
                    Bound::Upper
                } else if best >= beta {
                    Bound::Lower
                } else {
                    Bound::Exact
                },
                best: best_move,
                generation: 0,
            });
        }
        best
    }
    fn quiesce(
        &mut self,
        p: &mut Position,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        pv: &mut Vec<Move>,
    ) -> i32 {
        if !self.visit(true) {
            return 0;
        }
        let mut moves = p.legal_moves();
        let check = p.is_in_check(p.side_to_move());
        if moves.is_empty() {
            return if check { -MATE + ply as i32 } else { 0 };
        }
        if self.draw(p) {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.static_eval(p);
        }
        let mut best = if check { -INF } else { self.static_eval(p) };
        if !check {
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
            moves.retain(|&mv| Self::capture(p, mv) || mv.promotion.is_some());
        }
        self.ordered(p, &mut moves, None, ply);
        for mv in moves {
            let undo = p.make_move(mv);
            self.evaluator.after_make(p, mv);
            self.history.push(p.hash());
            let mut child = Vec::new();
            let score = -self.quiesce(p, ply + 1, -beta, -alpha, &mut child);
            self.history.pop();
            p.unmake_move(mv, undo);
            self.evaluator.after_unmake(p, mv);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                pv.clear();
                pv.push(mv);
                pv.extend(child);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        best
    }
}
fn to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE - MAX_PLY as i32 {
        score + ply as i32
    } else if score <= -MATE + MAX_PLY as i32 {
        score - ply as i32
    } else {
        score
    }
}
fn from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE - MAX_PLY as i32 {
        score - ply as i32
    } else if score <= -MATE + MAX_PLY as i32 {
        score + ply as i32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_mate_distance_rebases_at_another_ply() {
        assert_eq!(from_tt(to_tt(MATE - 7, 3), 5), MATE - 9);
        assert_eq!(from_tt(to_tt(-MATE + 7, 3), 5), -MATE + 9);
        for score in [-100, 0, 100] {
            assert_eq!(from_tt(to_tt(score, 3), 5), score);
        }
    }
}
