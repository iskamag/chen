#![cfg(all(feature = "hip", target_os = "linux"))]
use chen::{
    Position, SearchLimits,
    backend::SearchBackend,
    neural::{HipNeuralSelectiveTree, HipNeuralTree, preload_configured_runtime},
    resident::ResidentSearch,
    rules::RuleState,
    selective::{SelectiveConfig, SelectiveTree},
};
#[test]
#[ignore = "requires actual gfx1030, pinned inference archive/plugin and producing Torch runtime"]
fn actual_neural_selective_closed_horizon_and_failure_invalidation() {
    let _runtime = preload_configured_runtime()
        .unwrap()
        .expect("CHEN_NEURAL_MODEL required");
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    ] {
        let p = Position::from_fen(fen).unwrap();
        let mut reference = ResidentSearch::new(
            Box::new(HipNeuralTree::from_environment(5000).unwrap()),
            "neural",
        )
        .unwrap();
        let expected = reference
            .search(
                &mut p.clone(),
                SearchLimits {
                    depth: 2,
                    nodes: None,
                    time: None,
                },
                &[],
                &mut |_| {},
            )
            .unwrap();
        let mut tree = HipNeuralSelectiveTree::from_environment(5000).unwrap();
        let cfg = SelectiveConfig {
            max_ply: 2,
            batch_leaves: 8,
            ..SelectiveConfig::default()
        };
        tree.reset(RuleState::from(&p), &[], cfg).unwrap();
        for step in 0..100 {
            let stats = tree.advance(5000, 2).unwrap();
            assert!(stats.classified_positions <= 5000);
            if stats.closed() {
                break;
            }
            assert_eq!(stats.stop_reason, 0);
            assert!(step < 99);
        }
        let (score, pv, stats) = tree.finish().unwrap();
        assert!(stats.closed());
        assert_eq!((score, pv), (expected.score, expected.pv), "{fen}");
        // A caller error after publication must not permit a stale finish.
        assert!(tree.advance(5000, 1).is_err());
        assert!(tree.finish().is_err());
        tree.reset(RuleState::from(&p), &[], cfg).unwrap();
        assert!(tree.advance(5000, 2).unwrap().published());
        assert!(tree.finish().is_ok());
    }
}
