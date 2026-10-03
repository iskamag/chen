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
    /// Scout later moves and re-search only an improving, non-cutting probe.
    pub pvs: bool,
    /// Initial iterative-deepening half-window in cp; zero keeps full windows.
    pub aspiration_window: u16,
    pub hash_mb: usize,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            tt: true,
            quiescence: true,
            move_ordering: true,
            pvs: false,
            aspiration_window: 0,
            hash_mb: 16,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub pvs_probes: u64,
    pub pvs_researches: u64,
    pub aspiration_retries: u64,
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
    stats: SearchStats,
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
            stats: SearchStats::default(),
        }
    }
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }
    pub fn stats(&self) -> SearchStats {
        self.stats
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
        self.stats = SearchStats::default();
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
                let mut delta = if depth > 1 && self.config.aspiration_window > 0 {
                    i32::from(self.config.aspiration_window).min(INF)
                } else {
                    2 * INF
                };
                // Keep the previous completed score as the centre. Both sides
                // widen geometrically, eventually reaching the full window.
                // Failed windows and interrupted retries never publish a depth.
                let (score, pv) = loop {
                    let alpha = (result.score - delta).max(-INF);
                    let beta = (result.score + delta).min(INF);
                    let mut pv = Vec::new();
                    let score = self.negamax(p, depth as i32, 0, alpha, beta, &mut pv);
                    if self.aborted || (score > alpha && score < beta) {
                        break (score, pv);
                    }
                    self.stats.aspiration_retries += 1;
                    delta = (delta * 2).min(2 * INF);
                };
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
        #[cfg(feature = "causal-profile")]
        if self.nodes & 255 == 0 {
            coz::progress!("cpu_256_visited_nodes");
        }
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
        for (index, mv) in moves.into_iter().enumerate() {
            let quiet = !Self::capture(p, mv) && mv.promotion.is_none();
            let undo = p.make_move(mv);
            self.evaluator.after_make(p, mv);
            self.history.push(p.hash());
            let mut child = Vec::new();
            let score = if self.config.pvs && index > 0 && beta - alpha > 1 {
                self.stats.pvs_probes += 1;
                let probe = -self.negamax(p, depth - 1, ply + 1, -alpha - 1, -alpha, &mut child);
                if !self.aborted && probe > alpha && probe < beta {
                    self.stats.pvs_researches += 1;
                    child.clear();
                    -self.negamax(p, depth - 1, ply + 1, -beta, -alpha, &mut child)
                } else {
                    probe
                }
            } else {
                -self.negamax(p, depth - 1, ply + 1, -beta, -alpha, &mut child)
            };
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
    use crate::eval::{Material, Tapered};

    // Exhaustive finite-horizon oracle: no ordering, TT, PVS or alpha-beta.
    fn exhaustive(p: &mut Position, depth: i32, ply: i32) -> i32 {
        let moves = p.legal_moves();
        if moves.is_empty() {
            return if p.is_in_check(p.side_to_move()) {
                -MATE + ply
            } else {
                0
            };
        }
        if p.halfmove_clock() >= 100 || p.insufficient_material() {
            return 0;
        }
        if depth == 0 {
            return Material.evaluate(p);
        }
        moves
            .into_iter()
            .map(|mv| {
                let undo = p.make_move(mv);
                let value = -exhaustive(p, depth - 1, ply + 1);
                p.unmake_move(mv, undo);
                value
            })
            .max()
            .unwrap()
    }

    #[test]
    fn pvs_windows_match_exhaustive_bounds_and_original_tt_tags() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
        ] {
            let mut p = Position::from_fen(fen).unwrap();
            let before = p;
            let true_value = exhaustive(&mut p, 2, 0);
            for pvs in [false, true] {
                for ordering in [false, true] {
                    for (alpha, beta, bound) in [
                        (-INF, INF, Bound::Exact),
                        (true_value - 2, true_value - 1, Bound::Lower),
                        (true_value + 1, true_value + 2, Bound::Upper),
                        (true_value, true_value + 1, Bound::Upper),
                    ] {
                        let mut s = Searcher::new(
                            Box::new(Material),
                            SearchConfig {
                                pvs,
                                move_ordering: ordering,
                                quiescence: false,
                                hash_mb: 1,
                                ..SearchConfig::default()
                            },
                        );
                        s.history.push(p.hash());
                        let mut pv = Vec::new();
                        let value = s.negamax(&mut p, 2, 0, alpha, beta, &mut pv);
                        assert_eq!(p, before);
                        match bound {
                            Bound::Exact => assert_eq!(value, true_value),
                            Bound::Lower => assert!(beta <= value && value <= true_value),
                            Bound::Upper => assert!(true_value <= value && value <= alpha),
                        }
                        assert_eq!(s.table.get(s.tt_key(&p)).unwrap().bound, bound);
                        if bound == Bound::Exact {
                            let undo = p.make_move(pv[0]);
                            assert_eq!(-exhaustive(&mut p, 1, 1), true_value);
                            p.unmake_move(pv[0], undo);
                        }
                        for mv in pv {
                            assert!(p.legal_moves().contains(&mv));
                            p.make_move(mv);
                        }
                        p = before;
                    }
                }
            }
        }
    }

    #[test]
    fn aspiration_retry_abort_does_not_publish_failed_window() {
        let config = SearchConfig {
            pvs: true,
            aspiration_window: 1,
            hash_mb: 1,
            ..SearchConfig::default()
        };
        let limits = SearchLimits {
            depth: 1,
            nodes: None,
            time: None,
        };
        let mut p = Position::startpos();
        let before = p;
        // Determine the exact boundary after one completed failed aspiration
        // attempt. The next retry must hit its node limit before visiting a node.
        let mut scout = Searcher::new(Box::new(Tapered), config.clone());
        let first = scout.search(&mut p, limits, &[], |_| {});
        let mut pv = Vec::new();
        let value = scout.negamax(&mut p, 2, 0, first.score - 1, first.score + 1, &mut pv);
        assert!(value <= first.score - 1);
        let budget = scout.nodes;
        let mut s = Searcher::new(Box::new(Tapered), config);
        let mut publications = Vec::new();
        let stopped = s.search(
            &mut p,
            SearchLimits {
                depth: 3,
                nodes: Some(budget),
                time: None,
            },
            &[],
            |r| publications.push((r.depth, r.score, r.pv.clone())),
        );
        assert!(stopped.stopped);
        assert_eq!(stopped.nodes, budget);
        assert_eq!(
            (stopped.depth, stopped.score, stopped.pv),
            (first.depth, first.score, first.pv)
        );
        assert_eq!(publications.len(), 1);
        assert_eq!(s.stats.aspiration_retries, 1);
        assert_eq!(p, before);
        assert_eq!(s.history, vec![p.hash()]);
    }

    #[test]
    fn cached_mate_distance_rebases_at_another_ply() {
        assert_eq!(from_tt(to_tt(MATE - 7, 3), 5), MATE - 9);
        assert_eq!(from_tt(to_tt(-MATE + 7, 3), 5), -MATE + 9);
        for score in [-100, 0, 100] {
            assert_eq!(from_tt(to_tt(score, 3), 5), score);
        }
    }
}
