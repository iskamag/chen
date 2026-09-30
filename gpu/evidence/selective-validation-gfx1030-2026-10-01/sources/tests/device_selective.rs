#![cfg(all(feature = "hip", target_os = "linux"))]
use chen::{
    Move, Position,
    device::HipSelectiveTree,
    rules::RuleState,
    selective::{HostSelectiveTree, SelectiveConfig, SelectiveStats, SelectiveTree},
};
fn same(mut host: SelectiveStats, gpu: SelectiveStats) {
    assert!(gpu.kernel_ms.is_finite() && gpu.kernel_ms >= 0.0);
    host.kernel_ms = gpu.kernel_ms;
    assert_eq!(host, gpu);
}
fn legal_pv(mut p: Position, pv: &[Move]) {
    for &mv in pv {
        assert!(p.legal_moves().contains(&mv));
        p.make_move(mv);
    }
}
#[test]
#[ignore = "requires actual gfx1030 HIP; scripts/rocm_run.sh test"]
fn actual_selective_stepwise_arena_parity_and_complete_horizons() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
        "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1",
    ] {
        let p = Position::from_fen(fen).unwrap();
        let mut host = HostSelectiveTree::new(100_000).unwrap();
        let mut gpu = HipSelectiveTree::from_environment(100_000).unwrap();
        for evaluator in [0, 1] {
            for batch in [1, 8, 32] {
                let cfg = SelectiveConfig {
                    max_ply: 2,
                    batch_leaves: batch,
                    ..SelectiveConfig::default()
                };
                same(
                    host.reset(RuleState::from(&p), &[], cfg).unwrap(),
                    gpu.reset(RuleState::from(&p), &[], cfg).unwrap(),
                );
                for step in 0..1000 {
                    let a = host.advance(u64::MAX, evaluator).unwrap();
                    let b = gpu.advance(u64::MAX, evaluator).unwrap();
                    same(a, b);
                    if a.published() {
                        let a = host.finish().unwrap();
                        let b = gpu.finish().unwrap();
                        assert_eq!(
                            (a.0, &a.1),
                            (b.0, &b.1),
                            "{fen}: step{step}, batch{batch}, eval{evaluator}"
                        );
                        same(a.2, b.2);
                        legal_pv(p, &b.1);
                    }
                    if a.closed() {
                        break;
                    }
                    assert_eq!(a.stop_reason, 0);
                    assert!(step < 999, "finite tree did not close");
                }
            }
        }
    }
}
#[test]
#[ignore = "requires actual gfx1030 HIP; scripts/rocm_run.sh test"]
fn actual_selective_atomic_limits_history_invalidation_and_thread_transfer() {
    let root = Position::startpos();
    for capacity in [1, 20, 21, 41, 1000] {
        let mut host = HostSelectiveTree::new(capacity).unwrap();
        let mut gpu = HipSelectiveTree::from_environment(capacity).unwrap();
        for cap in [0, 1, 20, 21, 40, 41, 42, 99, 421] {
            let cfg = SelectiveConfig {
                max_ply: 3,
                batch_leaves: 1,
                candidate_limit: 1,
                ..SelectiveConfig::default()
            };
            same(
                host.reset(RuleState::from(&root), &[], cfg).unwrap(),
                gpu.reset(RuleState::from(&root), &[], cfg).unwrap(),
            );
            for step in 0..100 {
                let a = host.advance(cap, 1).unwrap();
                let b = gpu.advance(cap, 1).unwrap();
                same(a, b);
                assert!(b.classified_positions <= cap);
                if a.published() {
                    let a = host.finish().unwrap();
                    let b = gpu.finish().unwrap();
                    assert_eq!((a.0, a.1), (b.0, b.1));
                }
                if a.stop_reason != 0 {
                    break;
                }
                assert!(step < 99);
            }
        }
    }
    let mut p = root;
    let mut history = Vec::new();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        history.push(p.hash());
        let mv = p.parse_uci_move(text).unwrap();
        p.make_move(mv);
    }
    let mut gpu = HipSelectiveTree::from_environment(1000).unwrap();
    let mut host = HostSelectiveTree::new(1000).unwrap();
    let cfg = SelectiveConfig::default();
    same(
        host.reset(RuleState::from(&p), &history, cfg).unwrap(),
        gpu.reset(RuleState::from(&p), &history, cfg).unwrap(),
    );
    same(host.advance(1, 1).unwrap(), gpu.advance(1, 1).unwrap());
    assert_eq!(gpu.finish().unwrap().0, 0);
    gpu.reset(RuleState::from(&root), &[], cfg).unwrap();
    gpu.advance(21, 1).unwrap();
    let expected = gpu.finish().unwrap();
    let mut gpu = std::thread::spawn(move || {
        let got = gpu.finish().unwrap();
        assert_eq!((got.0, got.1), (expected.0, expected.1));
        gpu
    })
    .join()
    .unwrap();
    assert!(gpu.advance(21, 0).is_err());
    assert!(gpu.finish().is_err());
    gpu.reset(RuleState::from(&root), &[], cfg).unwrap();
    gpu.advance(21, 1).unwrap();
    assert!(gpu.advance(20, 1).is_err());
    assert!(gpu.finish().is_err());
    gpu.reset(RuleState::from(&root), &[], cfg).unwrap();
    assert!(gpu.advance(21, 1).unwrap().published());
}
