use chen::{
    Color, Evaluator, Material, Move, Position, SearchConfig, SearchLimits, Searcher, Tapered,
    search::MATE,
};
use std::{
    sync::{Arc, Mutex, atomic::Ordering},
    time::Duration,
};
fn search(fen: &str, depth: u8, config: SearchConfig) -> chen::SearchResult {
    let mut p = Position::from_fen(fen).unwrap();
    let before = p;
    let mut s = Searcher::new(Box::new(Tapered), config);
    let r = s.search(
        &mut p,
        SearchLimits {
            depth,
            nodes: None,
            time: None,
        },
        &[],
        |_| {},
    );
    assert_eq!(p, before, "search must restore every board field");
    r
}
#[test]
fn mate_and_stalemate() {
    let mate = search("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 3, SearchConfig::default());
    assert_eq!(mate.score, -MATE);
    assert_eq!(mate.best_move, None);
    let stale = search("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, SearchConfig::default());
    assert_eq!(stale.score, 0);
    assert_eq!(stale.best_move, None);
    let win = search("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", 3, SearchConfig::default());
    assert_eq!(win.score, MATE - 1);
    let mut p = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    p.make_move(win.best_move.unwrap());
    assert!(p.legal_moves().is_empty() && p.is_in_check(Color::Black));
}
#[test]
fn draw_clock_does_not_override_checkmate() {
    let r = search(
        "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1",
        2,
        SearchConfig::default(),
    );
    assert_eq!(r.score, -MATE);
    let r = search(
        "7k/8/8/8/8/8/4R3/K7 w - - 100 1",
        2,
        SearchConfig::default(),
    );
    assert_eq!(r.score, 0);
}
#[test]
fn repetition_and_tt_draw_context() {
    let mut p = Position::startpos();
    let mut history = Vec::new();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        history.push(p.hash());
        let mv = p.parse_uci_move(text).unwrap();
        p.make_move(mv);
    }
    let mut s = Searcher::new(Box::new(Tapered), SearchConfig::default());
    let r = s.search(
        &mut p,
        SearchLimits {
            depth: 3,
            nodes: None,
            time: None,
        },
        &history,
        |_| {},
    );
    assert_eq!(r.score, 0);
    let mut fresh = Position::startpos();
    let r = s.search(
        &mut fresh,
        SearchLimits {
            depth: 3,
            nodes: None,
            time: None,
        },
        &[],
        |_| {},
    );
    let reference = search(&fresh.to_fen(), 3, SearchConfig::default());
    assert_eq!(r.score, reference.score);
}
#[test]
fn full_width_ablations_preserve_scores() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
    ] {
        for q in [false, true] {
            let reference = search(
                fen,
                3,
                SearchConfig {
                    tt: false,
                    quiescence: q,
                    move_ordering: false,
                    hash_mb: 1,
                },
            );
            for (tt, ordering) in [(true, false), (false, true), (true, true)] {
                let r = search(
                    fen,
                    3,
                    SearchConfig {
                        tt,
                        quiescence: q,
                        move_ordering: ordering,
                        hash_mb: 1,
                    },
                );
                assert_eq!(
                    r.score, reference.score,
                    "tt={tt} ordering={ordering} q={q}"
                );
            }
        }
    }
}
#[test]
fn node_limit_and_stop_restore_position() {
    let mut p = Position::startpos();
    let before = p;
    let mut s = Searcher::new(Box::new(Tapered), SearchConfig::default());
    let r = s.search(
        &mut p,
        SearchLimits {
            depth: 20,
            nodes: Some(100),
            time: None,
        },
        &[],
        |_| {},
    );
    assert!(r.stopped);
    assert_eq!(r.nodes, 100);
    assert_eq!(p, before);
    assert!(p.legal_moves().contains(&r.best_move.unwrap()));
    s.stop_handle().store(true, Ordering::Relaxed);
    let r = s.search(&mut p, SearchLimits::default(), &[], |_| {});
    assert!(r.stopped);
    assert_eq!(r.nodes, 0);
    assert_eq!(p, before);
    s.stop_handle().store(false, Ordering::Relaxed);
    let r = s.search(
        &mut p,
        SearchLimits {
            depth: 64,
            nodes: None,
            time: Some(Duration::ZERO),
        },
        &[],
        |_| {},
    );
    assert!(r.stopped);
    assert_eq!(p, before);
}
#[test]
fn evaluator_orientation() {
    let p = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
    let b = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1").unwrap();
    assert_eq!(Material.evaluate(&p), 100);
    assert_eq!(Material.evaluate(&b), -100);
    assert_eq!(Tapered.evaluate(&p), -Tapered.evaluate(&b));
}
struct Tracking {
    stack: Vec<u64>,
    events: Arc<Mutex<(usize, usize)>>,
}
impl Evaluator for Tracking {
    fn name(&self) -> &'static str {
        "tracking"
    }
    fn reset(&mut self, p: &Position) {
        self.stack = vec![p.hash()];
    }
    fn evaluate(&mut self, p: &Position) -> i32 {
        assert_eq!(self.stack.last(), Some(&p.hash()));
        Material.evaluate(p)
    }
    fn after_make(&mut self, p: &Position, _: Move) {
        self.stack.push(p.hash());
        self.events.lock().unwrap().0 += 1;
    }
    fn after_unmake(&mut self, p: &Position, _: Move) {
        self.stack.pop();
        assert_eq!(self.stack.last(), Some(&p.hash()));
        self.events.lock().unwrap().1 += 1;
    }
}
#[test]
fn incremental_evaluator_hooks_are_balanced_on_abort() {
    let events = Arc::new(Mutex::new((0, 0)));
    let mut s = Searcher::new(
        Box::new(Tracking {
            stack: Vec::new(),
            events: events.clone(),
        }),
        SearchConfig::default(),
    );
    s.search(
        &mut Position::startpos(),
        SearchLimits {
            depth: 10,
            nodes: Some(500),
            time: None,
        },
        &[],
        |_| {},
    );
    let counts = *events.lock().unwrap();
    assert!(counts.0 > 0);
    assert_eq!(counts.0, counts.1);
}

// Exhaustive minimax oracle has no alpha-beta, ordering, TT or quiescence.
fn minimax(p: &mut Position, depth: u8, ply: i32) -> i32 {
    let moves = p.legal_moves();
    if moves.is_empty() {
        return if p.is_in_check(p.side_to_move()) {
            -MATE + ply
        } else {
            0
        };
    }
    if p.insufficient_material() || p.halfmove_clock() >= 100 {
        return 0;
    }
    if depth == 0 {
        return Material.evaluate(p);
    }
    let mut best = i32::MIN;
    for mv in moves {
        let undo = p.make_move(mv);
        best = best.max(-minimax(p, depth - 1, ply + 1));
        p.unmake_move(mv, undo);
    }
    best
}
#[test]
fn alpha_beta_matches_exhaustive_minimax() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
        "4k3/8/8/3q4/8/2N5/4P3/4K3 w - - 0 1",
    ] {
        let mut p = Position::from_fen(fen).unwrap();
        let expected = minimax(&mut p, 3, 0);
        let mut s = Searcher::new(
            Box::new(Material),
            SearchConfig {
                quiescence: false,
                ..SearchConfig::default()
            },
        );
        let result = s.search(
            &mut p,
            SearchLimits {
                depth: 3,
                nodes: None,
                time: None,
            },
            &[],
            |_| {},
        );
        assert_eq!(result.score, expected, "{fen}");
    }
}

#[test]
fn experiment_changes_invalidate_cached_scores() {
    let mut p = Position::startpos();
    let mut s = Searcher::new(Box::new(Tapered), SearchConfig::default());
    let limits = SearchLimits {
        depth: 3,
        nodes: None,
        time: None,
    };
    s.search(&mut p, limits.clone(), &[], |_| {});
    s.set_evaluator(Box::new(Material));
    s.set_config(SearchConfig {
        quiescence: false,
        hash_mb: 1,
        ..SearchConfig::default()
    });
    let result = s.search(&mut p, limits, &[], |_| {});
    assert_eq!(result.score, minimax(&mut p, 3, 0));
    assert_eq!(s.evaluator_name(), "material");
    assert_eq!(s.config().hash_mb, 1);
}

struct Extreme;
impl Evaluator for Extreme {
    fn name(&self) -> &'static str {
        "extreme"
    }
    fn evaluate(&mut self, _: &Position) -> i32 {
        i32::MAX
    }
}
#[test]
fn neural_or_custom_scores_cannot_enter_mate_range() {
    let mut s = Searcher::new(
        Box::new(Extreme),
        SearchConfig {
            quiescence: false,
            ..SearchConfig::default()
        },
    );
    let result = s.search(
        &mut Position::startpos(),
        SearchLimits {
            depth: 1,
            nodes: None,
            time: None,
        },
        &[],
        |_| {},
    );
    assert_eq!(result.score, -chen::search::MAX_EVAL);
    assert!(result.uci_info().contains("score cp -29000"));
}
