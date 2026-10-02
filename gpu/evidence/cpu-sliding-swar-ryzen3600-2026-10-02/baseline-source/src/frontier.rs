//! Full-width, level-synchronous minimax with batched leaf evaluation.
//! CPU rules are the oracle; batch inference can execute on a device.
//! This is a bounded research baseline, without quiescence, TT, or pruning.
use crate::{
    Move, Position, SearchLimits, SearchResult,
    backend::{BatchEvaluator, BatchTelemetry, SearchBackend},
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

#[derive(Clone, Debug)]
pub struct FrontierConfig {
    pub batch_size: usize,
    pub max_positions: usize,
}
impl Default for FrontierConfig {
    fn default() -> Self {
        Self {
            batch_size: 4096,
            max_positions: 250_000,
        }
    }
}
struct Node {
    position: Position,
    parent: Option<usize>,
    mv: Option<Move>,
    children: Range<usize>,
    score: Option<i32>,
    best: Option<usize>,
    ply: usize,
}
pub struct FrontierSearch {
    evaluator: Box<dyn BatchEvaluator>,
    config: FrontierConfig,
    stop: Arc<AtomicBool>,
    telemetry: BatchTelemetry,
}
impl FrontierSearch {
    pub fn new(evaluator: Box<dyn BatchEvaluator>, config: FrontierConfig) -> Result<Self, String> {
        if config.batch_size == 0 || config.max_positions == 0 {
            return Err("frontier capacities must be positive".into());
        }
        Ok(Self {
            evaluator,
            config,
            stop: Arc::new(AtomicBool::new(false)),
            telemetry: BatchTelemetry::default(),
        })
    }
    fn interrupted(&self, start: Instant, limits: &SearchLimits) -> bool {
        self.stop.load(Ordering::Relaxed) || limits.time.is_some_and(|cap| start.elapsed() >= cap)
    }
    fn repeated(tree: &[Node], mut index: usize, history: &[u64]) -> bool {
        let key = tree[index].position.hash();
        let mut remaining = tree[index].position.halfmove_clock() as usize + 1;
        let mut count = 0;
        loop {
            if remaining == 0 {
                break;
            }
            count += usize::from(tree[index].position.hash() == key);
            remaining -= 1;
            if let Some(parent) = tree[index].parent {
                index = parent;
            } else {
                break;
            }
        }
        count += history
            .iter()
            .rev()
            .take(remaining)
            .filter(|&&hash| hash == key)
            .count();
        count >= 3
    }
    fn iteration(
        &mut self,
        root: Position,
        depth: u8,
        history: &[u64],
        limits: &SearchLimits,
        start: Instant,
        nodes: &mut u64,
    ) -> Result<Option<(i32, Vec<Move>)>, String> {
        let mut tree = vec![Node {
            position: root,
            parent: None,
            mv: None,
            children: 0..0,
            score: None,
            best: None,
            ply: 0,
        }];
        let mut cursor = 0;
        let mut leaves = Vec::new();
        while cursor < tree.len() {
            if self.interrupted(start, limits) || limits.nodes.is_some_and(|cap| *nodes >= cap) {
                return Ok(None);
            }
            *nodes += 1;
            let position = tree[cursor].position;
            let moves = position.legal_moves();
            let ply = tree[cursor].ply;
            if moves.is_empty() {
                tree[cursor].score = Some(if position.is_in_check(position.side_to_move()) {
                    -MATE + ply as i32
                } else {
                    0
                });
            } else if position.halfmove_clock() >= 100
                || position.insufficient_material()
                || Self::repeated(&tree, cursor, history)
            {
                tree[cursor].score = Some(0);
            } else if ply == depth as usize {
                leaves.push(cursor);
            } else {
                if tree.len().saturating_add(moves.len()) > self.config.max_positions {
                    return Ok(None);
                }
                let from = tree.len();
                for mv in moves {
                    let mut child = position;
                    child.make_move(mv);
                    tree.push(Node {
                        position: child,
                        parent: Some(cursor),
                        mv: Some(mv),
                        children: 0..0,
                        score: None,
                        best: None,
                        ply: ply + 1,
                    });
                }
                tree[cursor].children = from..tree.len();
            }
            cursor += 1;
        }
        for batch in leaves.chunks(self.config.batch_size) {
            if self.interrupted(start, limits) {
                return Ok(None);
            }
            let positions: Vec<_> = batch.iter().map(|&i| tree[i].position).collect();
            let mut scores = vec![0; batch.len()];
            let telemetry = self.evaluator.evaluate_batch(&positions, &mut scores)?;
            self.telemetry.accumulate(telemetry);
            for (&index, score) in batch.iter().zip(scores) {
                tree[index].score = Some(score.clamp(-MAX_EVAL, MAX_EVAL));
            }
        }
        for index in (0..tree.len()).rev() {
            if index & 255 == 0 && self.interrupted(start, limits) {
                return Ok(None);
            }
            if !tree[index].children.is_empty() {
                let mut best_score = i32::MIN;
                let mut best_child = None;
                for child in tree[index].children.clone() {
                    let score = -tree[child].score.expect("child reduced first");
                    if score > best_score {
                        best_score = score;
                        best_child = Some(child);
                    }
                }
                tree[index].score = Some(best_score);
                tree[index].best = best_child;
            }
        }
        let mut pv = Vec::new();
        let mut index = 0;
        while let Some(child) = tree[index].best {
            pv.push(tree[child].mv.unwrap());
            index = child;
        }
        Ok(Some((tree[0].score.unwrap(), pv)))
    }
}
impl SearchBackend for FrontierSearch {
    fn name(&self) -> &str {
        "frontier"
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
        position: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        let start = Instant::now();
        self.telemetry = BatchTelemetry::default();
        let moves = position.legal_moves();
        let mut result = SearchResult {
            best_move: moves.first().copied(),
            score: if moves.is_empty() && position.is_in_check(position.side_to_move()) {
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
        if moves.is_empty() {
            result.elapsed = start.elapsed();
            return Ok(result);
        }
        let mut nodes = 0;
        for depth in 1..=limits.depth.clamp(1, 64) {
            match self.iteration(*position, depth, history, &limits, start, &mut nodes)? {
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
