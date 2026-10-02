use chen::{
    Position, SearchLimits,
    backend::{CpuBatchEvaluator, SearchBackend},
    eval::Tapered,
    frontier::{FrontierConfig, FrontierSearch},
    resident::{HostTree, ResidentSearch, ResidentTree, TreeError, TreeSpan},
    rules::RuleState,
};
use std::{sync::atomic::Ordering, time::Duration};
fn limits(depth: u8) -> SearchLimits {
    SearchLimits {
        depth,
        nodes: None,
        time: None,
    }
}
fn tree(cap: usize) -> ResidentSearch {
    ResidentSearch::new(Box::new(HostTree::new(cap).unwrap()), "tapered").unwrap()
}
#[test]
fn host_resident_matches_independent_frontier_full_results() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "7k/8/8/8/8/8/4R3/K7 w - - 100 1",
    ] {
        let mut p = Position::from_fen(fen).unwrap();
        let before = p;
        let mut resident = tree(250_000);
        let mut frontier = FrontierSearch::new(
            Box::new(CpuBatchEvaluator::new(Box::new(Tapered))),
            FrontierConfig::default(),
        )
        .unwrap();
        let a = resident
            .search(&mut p, limits(3), &[], &mut |_| {})
            .unwrap();
        let b = frontier
            .search(&mut p, limits(3), &[], &mut |_| {})
            .unwrap();
        assert_eq!(
            (a.depth, a.score, a.best_move, a.nodes, a.pv),
            (b.depth, b.score, b.best_move, b.nodes, b.pv)
        );
        assert_eq!(p, before);
    }
}
#[test]
fn resident_budget_and_overflow_preserve_completed_iterations() {
    let mut p = Position::startpos();
    let mut r = tree(22);
    let a = r.search(&mut p, limits(3), &[], &mut |_| {}).unwrap();
    assert_eq!(a.depth, 1);
    assert!(a.stopped);
    for cap in [0, 1, 10, 21] {
        let a = r
            .search(
                &mut p,
                SearchLimits {
                    depth: 1,
                    nodes: Some(cap),
                    time: None,
                },
                &[],
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(a.nodes, cap);
        assert_eq!(a.depth, u8::from(cap == 21));
        assert_eq!(a.stopped, cap != 21);
    }
    r.stop_handle().store(true, Ordering::Relaxed);
    assert_eq!(
        r.search(&mut p, limits(2), &[], &mut |_| {}).unwrap().nodes,
        0
    );
    r.stop_handle().store(false, Ordering::Relaxed);
    let mut l = limits(2);
    l.time = Some(Duration::ZERO);
    assert!(r.search(&mut p, l, &[], &mut |_| {}).unwrap().stopped);
}
#[test]
fn resident_history_and_phase_validation() {
    let mut p = Position::startpos();
    let mut history = Vec::new();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        history.push(p.hash());
        let mv = p.parse_uci_move(text).unwrap();
        p.make_move(mv);
    }
    assert_eq!(
        tree(100)
            .search(&mut p, limits(2), &history, &mut |_| {})
            .unwrap()
            .score,
        0
    );
    let mut arena = HostTree::new(21).unwrap();
    let span = TreeSpan { start: 0, count: 1 };
    assert!(arena.finish().is_err());
    arena
        .reset(RuleState::from(&Position::startpos()), &[])
        .unwrap();
    assert!(arena.reduce(span).is_err());
    let stats = arena.expand(span, true).unwrap();
    assert_eq!(stats.next_count, 20);
    assert!(arena.evaluate(span, 1).is_err());
    assert!(arena.expand(span, true).is_err());
    assert!(arena.classify(span, true).is_err());
    assert!(matches!(
        arena.expand(stats.next_span(), true),
        Err(TreeError::Capacity(_))
    ));
    let stats = arena.classify(stats.next_span(), true).unwrap();
    assert_eq!(stats.arena_positions, 21);
    assert!(arena.classify(stats.next_span(), false).is_err());
    let leaf = TreeSpan {
        start: 1,
        count: 20,
    };
    let a = arena.evaluate(leaf, 1).unwrap();
    let b = arena.evaluate(leaf, 1).unwrap();
    assert_eq!(a.evaluated_positions, b.evaluated_positions);
    assert!(arena.evaluate(leaf, 0).is_err());
    assert!(
        arena
            .reduce(TreeSpan {
                start: 0,
                count: 21
            })
            .is_err()
    );
    arena
        .reset(RuleState::from(&Position::startpos()), &[])
        .unwrap();
    arena.evaluate(span, 1).unwrap();
    assert!(arena.expand(span, true).is_err());
    assert!(arena.reduce(span).is_err());
}
