use chen::{
    Position, SearchConfig, SearchLimits, Searcher,
    backend::{BatchEvaluator, CpuBatchEvaluator, SearchBackend},
    eval::{Material, Tapered},
    frontier::{FrontierConfig, FrontierSearch},
    search::MATE,
};
use std::{sync::atomic::Ordering, time::Duration};
fn limits(depth: u8) -> SearchLimits {
    SearchLimits {
        depth,
        nodes: None,
        time: None,
    }
}
#[test]
fn frontier_matches_alpha_beta_without_selectivity() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
    ] {
        let mut p = Position::from_fen(fen).unwrap();
        let original = p;
        let mut alpha = Searcher::new(
            Box::new(Tapered),
            SearchConfig {
                quiescence: false,
                ..SearchConfig::default()
            },
        );
        let expected = alpha.search(&mut p, limits(3), &[], |_| {});
        let mut frontier = FrontierSearch::new(
            Box::new(CpuBatchEvaluator::new(Box::new(Tapered))),
            FrontierConfig {
                batch_size: 7,
                max_positions: 50_000,
            },
        )
        .unwrap();
        let actual = frontier
            .search(&mut p, limits(3), &[], &mut |_| {})
            .unwrap();
        assert_eq!(actual.score, expected.score, "{fen}");
        assert_eq!(p, original);
        assert!(p.legal_moves().contains(&actual.best_move.unwrap()));
    }
}
#[test]
fn frontier_capacity_stop_and_node_budget_preserve_last_iteration() {
    let mut p = Position::startpos();
    let original = p;
    let mut f = FrontierSearch::new(
        Box::new(CpuBatchEvaluator::new(Box::new(Material))),
        FrontierConfig {
            batch_size: 16,
            max_positions: 22,
        },
    )
    .unwrap();
    let r = f.search(&mut p, limits(3), &[], &mut |_| {}).unwrap();
    assert_eq!(r.depth, 1);
    assert!(r.stopped);
    assert_eq!(p, original);
    let mut l = limits(4);
    l.nodes = Some(10);
    let r = f.search(&mut p, l, &[], &mut |_| {}).unwrap();
    assert_eq!(r.nodes, 10);
    assert!(r.stopped);
    f.stop_handle().store(true, Ordering::Relaxed);
    let r = f.search(&mut p, limits(4), &[], &mut |_| {}).unwrap();
    assert_eq!(r.nodes, 0);
    assert!(r.stopped);
    f.stop_handle().store(false, Ordering::Relaxed);
    let mut l = limits(4);
    l.time = Some(Duration::ZERO);
    assert!(f.search(&mut p, l, &[], &mut |_| {}).unwrap().stopped);
}
#[test]
fn frontier_mate_draws_and_batch_contract() {
    let mut f = FrontierSearch::new(
        Box::new(CpuBatchEvaluator::new(Box::new(Material))),
        FrontierConfig::default(),
    )
    .unwrap();
    for (fen, score) in [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", -MATE),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),
        ("7k/8/8/8/8/8/4R3/K7 w - - 100 1", 0),
    ] {
        assert_eq!(
            f.search(
                &mut Position::from_fen(fen).unwrap(),
                limits(2),
                &[],
                &mut |_| {}
            )
            .unwrap()
            .score,
            score
        );
    }
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
        f.search(&mut p, limits(2), &history, &mut |_| {})
            .unwrap()
            .score,
        0
    );
    let mut batch = CpuBatchEvaluator::new(Box::new(Material));
    assert!(batch.evaluate_batch(&[p], &mut []).is_err());
    assert_eq!(batch.evaluate_batch(&[], &mut []).unwrap().positions, 0);
}

#[test]
fn exact_node_budget_finishes_already_admitted_leaf_work() {
    let mut p = Position::startpos();
    let mut f = FrontierSearch::new(
        Box::new(CpuBatchEvaluator::new(Box::new(Tapered))),
        FrontierConfig::default(),
    )
    .unwrap();
    let mut l = limits(1);
    l.nodes = Some(21);
    let exact = f.search(&mut p, l, &[], &mut |_| {}).unwrap();
    let reference = f.search(&mut p, limits(1), &[], &mut |_| {}).unwrap();
    assert_eq!(exact.nodes, 21);
    assert_eq!(exact.depth, 1);
    assert!(!exact.stopped);
    assert_eq!(
        (exact.score, exact.best_move, exact.pv),
        (reference.score, reference.best_move, reference.pv)
    );
}
