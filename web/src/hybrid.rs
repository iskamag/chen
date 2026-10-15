//! The playable hybrid: bounded reply proofs, then Chen PVS in the remaining budget.
use chen::codec::CodecPosition;
use chen::codec_policy::Preferences;
use chen::{
    Move, OpeningBook, Position, SearchConfig, SearchLimits, SearchResult, Searcher, Tapered,
};
use chen_relation_r1::{self as relation, Mode, Verdict};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

struct Candidate {
    mv: Move,
    score: i32,
    terminal: Verdict,
    allows_mate: Option<bool>,
}

pub(super) struct HybridSearch {
    inner: Searcher,
}

impl HybridSearch {
    pub fn new() -> Self {
        Self {
            inner: Searcher::new(
                Box::new(Tapered),
                SearchConfig {
                    pvs: true,
                    aspiration_window: 25,
                    ..Default::default()
                },
            ),
        }
    }

    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.inner.stop_handle()
    }

    pub fn set_book(&mut self, book: Option<Arc<OpeningBook>>, seed: u64, plies: usize) {
        self.inner.set_book(book, seed, plies);
    }

    pub fn search(
        &mut self,
        position: &mut Position,
        mut limits: SearchLimits,
        history: &[u64],
    ) -> SearchResult {
        let started = Instant::now();
        let budget = limits.time.unwrap_or(Duration::from_secs(1));
        let deadline = started
            + budget
                .saturating_sub(Duration::from_millis(35))
                .max(Duration::from_millis(1));
        let gate_deadline = deadline.min(started + Duration::from_millis(25));
        let stop = self.stop_handle();
        let candidates = candidates(position, history, &stop, gate_deadline);
        let Some(first) = candidates.first() else {
            return self.inner.search(position, limits, history, |_| {});
        };
        let mate = candidates.iter().find(|c| c.terminal == Verdict::Mate);
        let choice = mate.unwrap_or(first);
        let mut result = SearchResult {
            best_move: Some(choice.mv),
            score: choice.score,
            depth: u8::from(mate.is_some()),
            nodes: 0,
            qnodes: 0,
            elapsed: started.elapsed(),
            pv: vec![choice.mv],
            stopped: false,
            hashfull: 0,
            book_move: None,
            selective: None,
        };
        if mate.is_none() && !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
            limits.time = Some(deadline.saturating_duration_since(Instant::now()));
            let mut searched = self.inner.search(position, limits, history, |_| {});
            let safe_exists = candidates.iter().any(|c| c.allows_mate == Some(false));
            let accepted = searched
                .best_move
                .and_then(|mv| candidates.iter().find(|c| c.mv == mv))
                .is_some_and(|c| !safe_exists || c.allows_mate == Some(false));
            if !accepted {
                // The chosen fallback has a certificate; do not report the rejected PV/book.
                searched.best_move = result.best_move;
                searched.score = result.score;
                searched.depth = 0;
                searched.pv = result.pv;
                searched.book_move = None;
            }
            result = searched;
        }
        result.elapsed = started.elapsed();
        result.stopped |= stop.load(Ordering::Relaxed);
        result
    }
}

fn candidates(
    position: &Position,
    history: &[u64],
    stop: &AtomicBool,
    deadline: Instant,
) -> Vec<Candidate> {
    let codec = CodecPosition::from_position(position).relation();
    let preferences = Preferences::new(position, &codec);
    let mut choices: Vec<_> = codec
        .decisions()
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            preferences.score(i).map(|s| Candidate {
                mv: d.mv,
                score: s.score,
                terminal: Verdict::Unknown,
                allows_mate: None,
            })
        })
        .collect();
    choices.sort_by_key(|c| std::cmp::Reverse(c.score));
    if choices.is_empty() || stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
        return choices;
    }
    let (base, actions) = relation::codec_root_actions(position);
    let (labels, _) = relation::classify(&base, &actions, Mode::Stream, 4096);
    for c in &mut choices {
        c.terminal = labels
            .iter()
            .find(|l| l.mv == c.mv)
            .map_or(Verdict::Unknown, |l| l.verdict);
        let mut child = *position;
        child.make_move(c.mv);
        let draw = child.halfmove_clock() >= 100
            || child.insufficient_material()
            || history
                .iter()
                .rev()
                .take(child.halfmove_clock() as usize)
                .filter(|&&h| h == child.hash())
                .count()
                >= 2;
        if c.terminal == Verdict::Mate {
            c.score = 30_000;
        } else if c.terminal == Verdict::Stalemate || draw {
            c.score = 0;
        }
    }
    if !choices.iter().any(|c| c.terminal == Verdict::Mate) {
        for c in &mut choices {
            if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                break;
            }
            if c.terminal == Verdict::Stalemate {
                c.allows_mate = Some(false);
                continue;
            }
            let child = base.after(actions.iter().find(|a| a.mv == c.mv).unwrap());
            let replies = relation::legal_actions(&child);
            let checking = relation::checking_actions(&child, &replies);
            if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                break;
            }
            let (labels, _) = relation::classify(&child, &checking, Mode::Stream, 4096);
            if labels.iter().any(|l| l.verdict == Verdict::Mate) {
                c.allows_mate = Some(true);
            } else if labels.iter().all(|l| l.verdict != Verdict::Unknown) {
                c.allows_mate = Some(false);
            }
        }
    }
    choices.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.terminal == Verdict::Mate),
            match c.allows_mate {
                Some(false) => 0,
                None => 1,
                Some(true) => 2,
            },
            std::cmp::Reverse(c.score),
        )
    });
    choices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chooses_mate_for_both_colors_without_mutating_root() {
        for fen in [
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/6k1/5q2/7K b - - 0 1",
        ] {
            let mut p = Position::from_fen(fen).unwrap();
            let mut hybrid = HybridSearch::new();
            let result = hybrid.search(
                &mut p,
                SearchLimits {
                    time: Some(Duration::from_millis(100)),
                    ..Default::default()
                },
                &[],
            );
            assert_eq!(p.to_fen(), fen);
            p.make_move(result.best_move.unwrap());
            assert!(p.legal_moves().is_empty());
            assert!(p.is_in_check(p.side_to_move()));
        }
    }

    #[test]
    fn respects_search_limits_and_shares_cancellation() {
        let mut hybrid = HybridSearch::new();
        let mut p = Position::startpos();
        let result = hybrid.search(
            &mut p,
            SearchLimits {
                depth: 2,
                nodes: Some(128),
                time: Some(Duration::from_millis(100)),
            },
            &[],
        );
        assert!(result.depth <= 2 && result.nodes <= 128);
        assert!(p.legal_moves().contains(&result.best_move.unwrap()));
        hybrid.stop_handle().store(true, Ordering::Relaxed);
        let result = hybrid.search(&mut p, SearchLimits::default(), &[]);
        assert!(result.stopped);
        assert!(p.legal_moves().contains(&result.best_move.unwrap()));
    }
}
