//! A deliberately distracted deterministic rollout school, not alpha-beta.
//! Scores are heuristic root-player centipawns; sampled future mates are not proofs.
use crate::{
    Color, Move, Piece, Position, SearchLimits, SearchResult,
    backend::SearchBackend,
    eval::{Evaluator, Tapered},
    search::{MATE, MAX_EVAL},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Copy, Debug)]
pub struct DrowsyConfig {
    /// Centipawn drift required before a walker updates its remembered score.
    pub delta_cp: i32,
    /// Ignore nonterminal moves whose deterministic fish attention is below theta.
    /// Positive infinity intentionally ignores every nonterminal move.
    pub theta: f32,
    pub seed: u64,
    pub walkers: usize,
    pub max_plies: u8,
}
impl Default for DrowsyConfig {
    fn default() -> Self {
        Self {
            delta_cp: 80,
            theta: 0.55,
            seed: 0x4348_454e,
            walkers: 8,
            max_plies: 6,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrowsyStats {
    pub nodes: u64,
    pub ignored_boring: u64,
    /// Suppressed evaluation notices plus walkers finding no interesting move.
    pub fail_meow: u64,
    /// Sudden noticed score changes at least delta_cp.
    pub fail_zoomies: u64,
    pub noticed_drift: u64,
    pub deepest_ply: u8,
    pub peak_walkers: usize,
}
struct Walker {
    position: Position,
    pv: Vec<Move>,
    history: Vec<u64>,
    remembered: i32,
    attention: f32,
    asleep: bool,
}
struct Kitten {
    position: Position,
    mv: Move,
    score: i32,
    attention: f32,
    terminal: bool,
}
pub struct DrowsySearch {
    config: DrowsyConfig,
    stop: Arc<AtomicBool>,
    stats: DrowsyStats,
}
impl DrowsySearch {
    pub fn new(config: DrowsyConfig) -> Result<Self, String> {
        if config.delta_cp < 0
            || config.theta.is_nan()
            || config.theta < 0.0
            || !(1..=32).contains(&config.walkers)
            || !(1..=32).contains(&config.max_plies)
        {
            return Err("Drowsy requires delta>=0, theta>=0, 1..32 walkers and plies".into());
        }
        Ok(Self {
            config,
            stop: Arc::new(AtomicBool::new(false)),
            stats: DrowsyStats::default(),
        })
    }
    pub fn stats(&self) -> DrowsyStats {
        self.stats
    }
    pub fn config(&self) -> DrowsyConfig {
        self.config
    }
    fn interrupted(&self, start: Instant, limits: &SearchLimits) -> bool {
        self.stop.load(Ordering::Relaxed)
            || limits.time.is_some_and(|time| start.elapsed() >= time)
            || limits.nodes.is_some_and(|nodes| self.stats.nodes >= nodes)
    }
    fn root_score(position: &Position, root: Color) -> i32 {
        let score = Tapered.evaluate(position).clamp(-MAX_EVAL, MAX_EVAL);
        if position.side_to_move() == root {
            score
        } else {
            -score
        }
    }
    /// Input history excludes the position itself, and includes its immediate parent.
    fn terminal(
        position: &Position,
        moves: &[Move],
        history: &[u64],
        root: Color,
        ply: u8,
    ) -> Option<i32> {
        if moves.is_empty() {
            let score = if position.is_in_check(position.side_to_move()) {
                -MATE + i32::from(ply)
            } else {
                0
            };
            return Some(if position.side_to_move() == root {
                score
            } else {
                -score
            });
        }
        if position.halfmove_clock() >= 100
            || position.insufficient_material()
            || history
                .iter()
                .rev()
                .take(position.halfmove_clock() as usize)
                .filter(|&&hash| hash == position.hash())
                .count()
                >= 2
        {
            return Some(0);
        }
        None
    }
    fn mix(mut value: u64) -> u64 {
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    fn fish(&self, parent: &Position, child: &Position, mv: Move, ply: u8) -> f32 {
        let entropy = Self::mix(
            self.config.seed
                ^ parent.hash()
                ^ (u64::from(mv.from) << 8)
                ^ (u64::from(mv.to) << 16)
                ^ ((mv.promotion.map_or(0, |p| p as u64 + 1)) << 24)
                ^ (u64::from(ply) << 32),
        );
        let mut lure = 0.08 + (entropy >> 40) as f32 / (1u32 << 24) as f32 * 0.35;
        let piece = parent.piece_at(mv.from).map(|(_, piece)| piece);
        if piece == Some(Piece::Knight) {
            lure += 0.25;
        }
        let ep = piece == Some(Piece::Pawn) && parent.en_passant_square() == Some(mv.to);
        if parent.piece_at(mv.to).is_some() || ep {
            lure += 0.20;
        }
        if mv.promotion.is_some() {
            lure += 0.35;
        }
        if child.is_in_check(child.side_to_move()) {
            lure += 0.18;
        }
        let forward = if parent.side_to_move() == Color::White {
            mv.to / 8 > mv.from / 8
        } else {
            mv.to / 8 < mv.from / 8
        };
        if forward {
            lure += 0.10;
        }
        lure.min(1.0)
    }
    /// Every visited child is terminal-classified before attention pruning.
    fn kitten(
        &mut self,
        parent: &Position,
        mv: Move,
        history: &[u64],
        root: Color,
        ply: u8,
    ) -> Kitten {
        let mut position = *parent;
        position.make_move(mv);
        self.stats.nodes += 1;
        self.stats.deepest_ply = self.stats.deepest_ply.max(ply);
        let moves = position.legal_moves();
        let terminal = Self::terminal(&position, &moves, history, root, ply);
        let attention = if let Some(score) = terminal {
            // Cats of either side still recognize an immediate mate or draw.
            let mover_score = if parent.side_to_move() == root {
                score
            } else {
                -score
            };
            if mover_score > MAX_EVAL { 2.0 } else { 1.1 }
        } else {
            self.fish(parent, &position, mv, ply)
        };
        let score = terminal.unwrap_or_else(|| {
            if attention >= self.config.theta {
                Self::root_score(&position, root)
            } else {
                // This child is discarded; don't evaluate a fish already ignored.
                0
            }
        });
        Kitten {
            position,
            mv,
            score,
            attention,
            terminal: terminal.is_some(),
        }
    }
    fn worthy(&mut self, kitten: &Kitten) -> bool {
        if kitten.terminal || kitten.attention >= self.config.theta {
            true
        } else {
            self.stats.ignored_boring += 1;
            false
        }
    }
}
impl SearchBackend for DrowsySearch {
    fn name(&self) -> &str {
        "drowsy-toy"
    }
    fn drowsy_stats(&self) -> Option<DrowsyStats> {
        Some(self.stats)
    }
    fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    fn clear(&mut self) {
        self.stats = DrowsyStats::default();
    }
    fn search(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        let start = Instant::now();
        self.stats = DrowsyStats::default();
        // Mandatory root bootstrap permits a legal fallback even at a zero budget.
        let root = position.side_to_move();
        let moves = position.legal_moves();
        let terminal = Self::terminal(position, &moves, history, root, 0);
        let mut result = SearchResult {
            best_move: moves.first().copied(),
            score: terminal.unwrap_or(0),
            depth: 0,
            nodes: 0,
            qnodes: 0,
            elapsed: start.elapsed(),
            pv: moves.first().copied().into_iter().collect(),
            stopped: false,
            hashfull: 0,
            book_move: None,
            selective: None,
        };
        if terminal.is_some() {
            report(&result);
            return Ok(result);
        }
        if limits.depth == 0 || self.interrupted(start, &limits) {
            result.stopped = true;
            result.elapsed = start.elapsed();
            return Ok(result);
        }
        self.stats.nodes = 1;
        let mut path_history: Vec<_> = history
            .iter()
            .rev()
            .take(position.halfmove_clock() as usize)
            .copied()
            .collect();
        path_history.reverse();
        path_history.push(position.hash());
        let mut walkers = Vec::with_capacity(self.config.walkers);
        for mv in moves {
            if self.interrupted(start, &limits) {
                result.stopped = true;
                break;
            }
            let kitten = self.kitten(position, mv, &path_history, root, 1);
            if kitten.terminal && kitten.score == MATE - 1 {
                result.best_move = Some(mv);
                result.pv = vec![mv];
                result.score = kitten.score;
                result.depth = 0;
                result.nodes = self.stats.nodes;
                result.elapsed = start.elapsed();
                report(&result);
                return Ok(result);
            }
            if !self.worthy(&kitten) {
                continue;
            }
            walkers.push(Walker {
                position: kitten.position,
                pv: vec![kitten.mv],
                history: path_history.clone(),
                remembered: kitten.score.clamp(-MAX_EVAL, MAX_EVAL),
                attention: kitten.attention,
                asleep: kitten.terminal,
            });
            walkers.sort_by(|a, b| b.attention.total_cmp(&a.attention));
            if walkers.len() > self.config.walkers {
                walkers.pop();
            }
            self.stats.peak_walkers = self.stats.peak_walkers.max(walkers.len());
        }
        self.stats.peak_walkers = walkers.len();
        let max_plies = limits.depth.min(self.config.max_plies);
        for ply in 2..=max_plies {
            for walker in &mut walkers {
                if self.interrupted(start, &limits) {
                    result.stopped = true;
                    break;
                }
                if walker.asleep {
                    continue;
                }
                let moves = walker.position.legal_moves();
                let mut child_history = walker.history.clone();
                child_history.push(walker.position.hash());
                let mut chosen: Option<Kitten> = None;
                for mv in moves {
                    if self.interrupted(start, &limits) {
                        result.stopped = true;
                        break;
                    }
                    let kitten = self.kitten(&walker.position, mv, &child_history, root, ply);
                    if self.worthy(&kitten)
                        && chosen
                            .as_ref()
                            .is_none_or(|old| kitten.attention > old.attention)
                    {
                        chosen = Some(kitten);
                    }
                }
                if let Some(kitten) = chosen {
                    walker.position = kitten.position;
                    walker.pv.push(kitten.mv);
                    walker.history = child_history;
                    let drift = (i64::from(kitten.score) - i64::from(walker.remembered)).abs();
                    if kitten.terminal || drift >= i64::from(self.config.delta_cp) {
                        walker.remembered = kitten.score.clamp(-MAX_EVAL, MAX_EVAL);
                        self.stats.noticed_drift += 1;
                        self.stats.fail_zoomies +=
                            u64::from(drift >= i64::from(self.config.delta_cp));
                    } else {
                        self.stats.fail_meow += 1;
                    }
                    walker.asleep = kitten.terminal;
                } else {
                    walker.asleep = true;
                    self.stats.fail_meow += 1;
                }
            }
            if result.stopped || walkers.iter().all(|walker| walker.asleep) {
                break;
            }
        }
        if let Some(winner) = walkers.iter().max_by(|a, b| {
            a.remembered
                .cmp(&b.remembered)
                .then_with(|| a.attention.total_cmp(&b.attention))
        }) {
            result.best_move = winner.pv.first().copied();
            result.pv = winner.pv.clone();
            result.score = winner.remembered;
        } else {
            self.stats.fail_meow += 1;
        }
        // No complete minimax horizon: depth zero, with explored ply available in stats.
        result.nodes = self.stats.nodes;
        result.elapsed = start.elapsed();
        result.stopped |= self.interrupted(start, &limits);
        report(&result);
        Ok(result)
    }
}
