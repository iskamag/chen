#![cfg(all(feature = "hip", target_os = "linux"))]
use chen::{
    Move, Position, SearchLimits,
    backend::SearchBackend,
    device::HipRules,
    resident::{HostTree, ResidentSearch, ResidentTree, TreeError, TreeSpan},
    rules::{CpuRules, RuleState, RulesBackend},
};
use std::{sync::atomic::Ordering, time::Duration};
fn fixtures() -> Vec<Position> {
    let mut positions = include_str!("../benches/positions.txt")
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .filter_map(|line| line.split_once('|'))
        .map(|(_, fen)| Position::from_fen(fen.trim()).unwrap())
        .collect::<Vec<_>>();
    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 65535 4294967295",
        "7k/8/5KQ1/8/8/8/8/8 b - - 100 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1",
    ] {
        positions.push(Position::from_fen(fen).unwrap());
    }
    let mut p = Position::startpos();
    let mut random = 0x1030_c4e5_u64;
    for _ in 0..512 {
        let moves = p.legal_moves();
        if moves.is_empty() || p.halfmove_clock() >= 100 {
            p = Position::startpos();
        } else {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            p.make_move(moves[random as usize % moves.len()]);
        }
        positions.push(p);
    }
    positions
}
#[test]
#[ignore = "requires real gfx1030 HIP; scripts/rocm_run.sh test"]
fn actual_rules_and_every_legal_transition_match_rust() {
    let mut gpu = HipRules::from_environment(20_000).unwrap();
    let mut cpu = CpuRules;
    let states = fixtures().iter().map(RuleState::from).collect::<Vec<_>>();
    for size in [0, 1, 7, 64, states.len()] {
        assert_eq!(
            gpu.legal_batch(&states[..size]).unwrap(),
            cpu.legal_batch(&states[..size]).unwrap()
        );
    }
    let legal = cpu.legal_batch(&states).unwrap();
    let mut roots = Vec::new();
    let mut moves = Vec::new();
    for (state, legal) in states.iter().zip(&legal.moves) {
        for &mv in legal {
            roots.push(*state);
            moves.push(mv);
        }
    }
    assert_eq!(
        gpu.make_batch(&roots, &moves).unwrap(),
        cpu.make_batch(&roots, &moves).unwrap()
    );
    assert!(gpu.make_batch(&states[..1], &[]).is_err());
    assert!(
        gpu.make_batch(&states[..1], &[Move::new(0, 63, None)])
            .is_err()
    );
    for field in 0..7 {
        let mut bad = states[0];
        match field {
            0 => bad.side = 2,
            1 => bad.castling = 16,
            2 => bad.ep_square = 64,
            3 => bad.halfmove = 65536,
            4 => bad.fullmove = 0,
            5 => bad.reserved = 1,
            _ => bad.hash ^= 1,
        }
        assert!(gpu.legal_batch(&[bad]).is_err());
        assert!(gpu.make_batch(&[bad], &[moves[0]]).is_err());
    }
    std::thread::spawn(move || {
        gpu.legal_batch(&states[..1]).unwrap();
    })
    .join()
    .unwrap();
}
#[test]
#[ignore = "requires real gfx1030 HIP; scripts/rocm_run.sh test"]
fn actual_resident_perft_special_moves_and_draws() {
    let mut gpu = HipRules::from_environment(250_000).unwrap();
    for (i, mut p) in fixtures()[..15].iter().copied().enumerate() {
        for depth in 0..=if i == 0 { 4 } else { 3 } {
            let expected = chen::perft(&mut p, depth);
            let (actual, stats) = gpu.perft(&RuleState::from(&p), depth).unwrap();
            assert_eq!(actual, expected, "{} depth {depth}", p.to_fen());
            assert!(stats.kernel_ms.is_finite() && stats.kernel_ms >= 0.0);
        }
    }
    let mut bounded = HipRules::from_environment(21).unwrap();
    let root = RuleState::from(&Position::startpos());
    assert!(matches!(
        bounded.perft(&root, 2),
        Err(TreeError::Capacity(_))
    ));
    assert_eq!(bounded.perft(&root, 1).unwrap().0, 20);
}
fn compare_search(p: Position, history: &[u64], capacity: usize, limits: SearchLimits) {
    let mut gpu = ResidentSearch::new(
        Box::new(HipRules::from_environment(capacity).unwrap()),
        "tapered",
    )
    .unwrap();
    let mut cpu =
        ResidentSearch::new(Box::new(HostTree::new(capacity).unwrap()), "tapered").unwrap();
    let mut position = p;
    let a = gpu
        .search(&mut position, limits.clone(), history, &mut |_| {})
        .unwrap();
    let b = cpu
        .search(&mut position, limits, history, &mut |_| {})
        .unwrap();
    assert_eq!(
        (a.depth, a.score, a.best_move, a.nodes, a.pv, a.stopped),
        (b.depth, b.score, b.best_move, b.nodes, b.pv, b.stopped),
        "{}",
        p.to_fen()
    );
    assert_eq!(position, p);
}
#[test]
#[ignore = "requires real gfx1030 HIP; scripts/rocm_run.sh test"]
fn actual_resident_search_matches_host_and_exact_budgets() {
    for p in fixtures()[..15].iter().copied() {
        compare_search(
            p,
            &[],
            250_000,
            SearchLimits {
                depth: 3,
                nodes: None,
                time: None,
            },
        );
    }
    for cap in [0, 1, 10, 21, 22, 441, 442] {
        compare_search(
            Position::startpos(),
            &[],
            500,
            SearchLimits {
                depth: 2,
                nodes: Some(cap),
                time: None,
            },
        );
    }
    compare_search(
        Position::startpos(),
        &[],
        22,
        SearchLimits {
            depth: 3,
            nodes: None,
            time: None,
        },
    );
    let mut p = Position::startpos();
    let mut history = Vec::new();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        history.push(p.hash());
        let mv = p.parse_uci_move(text).unwrap();
        p.make_move(mv);
    }
    compare_search(
        p,
        &history,
        22,
        SearchLimits {
            depth: 3,
            nodes: None,
            time: None,
        },
    );
    let mut search = ResidentSearch::new(
        Box::new(HipRules::from_environment(21).unwrap()),
        "material",
    )
    .unwrap();
    search.stop_handle().store(true, Ordering::Relaxed);
    assert_eq!(
        search
            .search(
                &mut p,
                SearchLimits {
                    depth: 3,
                    nodes: None,
                    time: None
                },
                &[],
                &mut |_| {}
            )
            .unwrap()
            .nodes,
        0
    );
    search.stop_handle().store(false, Ordering::Relaxed);
    assert!(
        search
            .search(
                &mut p,
                SearchLimits {
                    depth: 3,
                    nodes: None,
                    time: Some(Duration::ZERO)
                },
                &[],
                &mut |_| {}
            )
            .unwrap()
            .stopped
    );
}
#[test]
#[ignore = "requires real gfx1030 HIP; scripts/rocm_run.sh test"]
fn actual_resident_phase_and_download_contracts() {
    let mut gpu = HipRules::from_environment(21).unwrap();
    let root = RuleState::from(&Position::startpos());
    let span = TreeSpan { start: 0, count: 1 };
    assert!(gpu.finish().is_err());
    gpu.reset(root, &[]).unwrap();
    assert!(gpu.reduce(span).is_err());
    let stats = gpu.expand(span, true).unwrap();
    assert_eq!(stats.next_count, 20);
    let children = gpu.download(stats.next_span()).unwrap();
    let mut cpu = CpuRules;
    let moves = cpu.legal_batch(&[root]).unwrap().moves.remove(0);
    assert_eq!(children, cpu.make_batch(&vec![root; 20], &moves).unwrap());
    assert!(gpu.evaluate(span, 1).is_err());
    assert!(gpu.expand(span, true).is_err());
    assert!(gpu.classify(span, true).is_err());
    assert!(gpu.classify(stats.next_span(), false).is_err());
    assert!(matches!(
        gpu.expand(stats.next_span(), true),
        Err(TreeError::Capacity(_))
    ));
    let a = gpu.evaluate(stats.next_span(), 1).unwrap();
    let b = gpu.evaluate(stats.next_span(), 1).unwrap();
    assert_eq!(a.evaluated_positions, b.evaluated_positions);
    assert_eq!(a.classified_positions, b.classified_positions);
    assert!(gpu.evaluate(stats.next_span(), 0).is_err());
    assert!(
        gpu.reduce(TreeSpan {
            start: 0,
            count: 21
        })
        .is_err()
    );
    gpu.reduce(span).unwrap();
    gpu.finish().unwrap();
    assert!(
        gpu.download(TreeSpan {
            start: u64::MAX,
            count: 1
        })
        .is_err()
    );
    gpu.legal_batch(&[root]).unwrap();
    assert!(gpu.finish().is_err());
    gpu.reset(root, &[]).unwrap();
    assert!(gpu.reset(root, &vec![0; 65536]).is_err());
    gpu.reset(root, &[]).unwrap();
    gpu.evaluate(span, 1).unwrap();
    assert!(gpu.expand(span, true).is_err());
    assert!(gpu.reduce(span).is_err());
}
