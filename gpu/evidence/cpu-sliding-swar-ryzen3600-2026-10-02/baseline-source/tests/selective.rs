use chen::{
    Position, SearchLimits,
    backend::{CpuBatchEvaluator, SearchBackend},
    eval::Tapered,
    frontier::{FrontierConfig, FrontierSearch},
    rules::RuleState,
    selective::{
        CAPACITY_LIMIT, HORIZON_COMPLETE, HostSelectiveTree, NODE_LIMIT, PUBLISHED,
        SelectiveConfig, SelectiveSearch, SelectiveTree, TERMINAL,
    },
};
use std::{sync::atomic::Ordering, time::Duration};
fn config(ply: u32, batch: u32) -> SelectiveConfig {
    SelectiveConfig {
        max_ply: ply,
        batch_leaves: batch,
        ..SelectiveConfig::default()
    }
}
fn search(capacity: usize, batch: u32) -> SelectiveSearch {
    SelectiveSearch::new(
        Box::new(HostSelectiveTree::new(capacity).unwrap()),
        "tapered",
        config(3, batch),
    )
    .unwrap()
}
#[test]
fn closed_sparse_tree_matches_independent_full_width_scores_and_pvs() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ] {
        let root = Position::from_fen(fen).unwrap();
        let mut full = FrontierSearch::new(
            Box::new(CpuBatchEvaluator::new(Box::new(Tapered))),
            FrontierConfig::default(),
        )
        .unwrap();
        let expected = full
            .search(
                &mut root.clone(),
                SearchLimits {
                    depth: 2,
                    nodes: None,
                    time: None,
                },
                &[],
                &mut |_| {},
            )
            .unwrap();
        for batch in [1, 8, 32] {
            let mut arena = HostSelectiveTree::new(100_000).unwrap();
            arena
                .reset(RuleState::from(&root), &[], config(2, batch))
                .unwrap();
            let mut prior_nodes = 0;
            loop {
                let s = arena.advance(u64::MAX, 1).unwrap();
                assert!(s.classified_positions >= prior_nodes);
                assert_eq!(s.classified_positions, s.arena_positions);
                prior_nodes = s.classified_positions;
                if s.closed() {
                    break;
                }
                assert_eq!(s.stop_reason, 0);
            }
            let (score, pv, stats) = arena.finish().unwrap();
            assert_eq!(
                (score, pv),
                (expected.score, expected.pv.clone()),
                "{fen} batch{batch}"
            );
            assert_eq!(stats.maximum_frontier_depth, 2);
            assert_eq!(stats.open_frontier_leaves, 0);
        }
    }
}
#[test]
fn atomic_siblings_respect_node_capacity_and_retain_last_publication() {
    let root = Position::startpos();
    for cap in [0, 1, 10, 20, 21, 40, 41, 421] {
        let mut arena = HostSelectiveTree::new(1000).unwrap();
        arena
            .reset(RuleState::from(&root), &[], config(2, 8))
            .unwrap();
        loop {
            let s = arena.advance(cap, 1).unwrap();
            assert!(s.classified_positions <= cap);
            assert_eq!(
                s.classified_positions,
                if cap == 0 { 0 } else { s.arena_positions }
            );
            if s.stop_reason != 0 {
                assert_eq!(s.published(), cap >= 21);
                assert_eq!(s.closed(), cap >= 421);
                if cap < 421 {
                    assert_eq!(s.stop_reason, NODE_LIMIT);
                }
                break;
            }
        }
        if cap < 21 {
            assert!(arena.finish().is_err());
        } else {
            let (_, pv, s) = arena.finish().unwrap();
            assert!(!pv.is_empty());
            assert_eq!(
                s.arena_positions,
                if cap < 41 {
                    21
                } else if cap < 421 {
                    41
                } else {
                    421
                }
            );
        }
    }
    let mut limited = HostSelectiveTree::new(22).unwrap();
    limited
        .reset(RuleState::from(&root), &[], config(3, 8))
        .unwrap();
    let first = limited.advance(1000, 1).unwrap();
    assert!(first.published());
    let before = limited.finish().unwrap();
    let blocked = limited.advance(1000, 1).unwrap();
    assert_eq!(blocked.stop_reason, CAPACITY_LIMIT);
    let after = limited.finish().unwrap();
    assert_eq!((before.0, before.1), (after.0, after.1));
    assert_eq!(after.2.arena_positions, 21);
}
#[test]
fn closed_best_child_does_not_block_remaining_eligible_branches() {
    let p = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let mut arena = HostSelectiveTree::new(100_000).unwrap();
    arena.reset(RuleState::from(&p), &[], config(3, 1)).unwrap();
    let first = arena.advance(u64::MAX, 1).unwrap();
    assert!(first.open_frontier_leaves > 0);
    let pv = arena.finish().unwrap().1;
    let mut mate = p;
    mate.make_move(pv[0]);
    assert!(mate.legal_moves().is_empty() && mate.is_in_check(mate.side_to_move()));
    let second = arena.advance(u64::MAX, 1).unwrap();
    assert!(second.arena_positions > first.arena_positions);
    assert_eq!(second.completed_steps, first.completed_steps + 1);
    for _ in 0..10_000 {
        if arena.advance(u64::MAX, 1).unwrap().closed() {
            return;
        }
    }
    panic!("finite tree did not close");
}
#[test]
fn reset_phase_terminal_history_and_cancellation_are_explicit() {
    let mut arena = HostSelectiveTree::new(100).unwrap();
    assert!(arena.finish().is_err());
    let p = Position::startpos();
    arena.reset(RuleState::from(&p), &[], config(2, 1)).unwrap();
    arena.advance(21, 1).unwrap();
    assert!(arena.advance(21, 0).is_err());
    assert!(arena.finish().is_err());
    let mut invalid = config(2, 1);
    invalid.reserved = 1;
    assert!(arena.reset(RuleState::from(&p), &[], invalid).is_err());
    assert!(arena.finish().is_err());
    for fen in [
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "7k/8/8/8/8/8/4R3/K7 w - - 100 1",
    ] {
        let p = Position::from_fen(fen).unwrap();
        arena.reset(RuleState::from(&p), &[], config(2, 1)).unwrap();
        let stats = arena.advance(1, 1).unwrap();
        assert_eq!(
            stats.root_flags & (PUBLISHED | TERMINAL | HORIZON_COMPLETE),
            PUBLISHED | TERMINAL | HORIZON_COMPLETE
        );
        assert_eq!(stats.evaluated_positions, 0);
        assert!(arena.finish().unwrap().1.is_empty());
    }
    let mut repeated = p;
    let mut history = Vec::new();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        history.push(repeated.hash());
        let mv = repeated.parse_uci_move(text).unwrap();
        repeated.make_move(mv);
    }
    arena
        .reset(RuleState::from(&repeated), &history, config(2, 1))
        .unwrap();
    assert_eq!(arena.advance(1, 1).unwrap().root_flags & TERMINAL, TERMINAL);
    let mut s = search(1000, 8);
    s.stop_handle().store(true, Ordering::Relaxed);
    let result = s
        .search(&mut p.clone(), SearchLimits::default(), &[], &mut |_| {})
        .unwrap();
    assert_eq!(result.nodes, 0);
    assert!(result.stopped);
    s.stop_handle().store(false, Ordering::Relaxed);
    let result = s
        .search(
            &mut p.clone(),
            SearchLimits {
                time: Some(Duration::ZERO),
                ..SearchLimits::default()
            },
            &[],
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(result.nodes, 0);
    assert!(result.stopped);
}
#[test]
fn partial_horizon_reports_minimum_and_selective_depth_separately() {
    let mut p = Position::startpos();
    let mut s = search(1000, 1);
    let result = s
        .search(
            &mut p,
            SearchLimits {
                depth: 3,
                nodes: Some(41),
                time: None,
            },
            &[],
            &mut |_| {},
        )
        .unwrap();
    let stats = result.selective.unwrap();
    assert_eq!(result.depth, 1);
    assert_eq!(
        (stats.minimum_frontier_depth, stats.maximum_frontier_depth),
        (1, 2)
    );
    assert!(result.uci_info().contains("depth 1 seldepth 2"));
    assert!(result.stopped);
    assert_eq!(result.nodes, 41);
    assert_eq!(s.batch_telemetry().positions, 40);
}

#[test]
fn independently_versioned_config_and_stats_layout() {
    use chen::selective::SelectiveStats;
    use std::mem::{offset_of, size_of};
    assert_eq!(size_of::<SelectiveConfig>(), 32);
    assert_eq!(offset_of!(SelectiveConfig, reserved), 24);
    assert_eq!(size_of::<SelectiveStats>(), 96);
    assert_eq!(offset_of!(SelectiveStats, arena_positions), 8);
    assert_eq!(offset_of!(SelectiveStats, last_selected), 56);
    assert_eq!(offset_of!(SelectiveStats, kernel_ms), 64);
    assert_eq!(offset_of!(SelectiveStats, minimum_frontier_depth), 72);
    let stats = SelectiveStats::empty();
    assert_eq!((stats.abi_version, stats.struct_size), (1, 96));
}
