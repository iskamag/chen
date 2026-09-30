#![cfg(all(feature = "hip", target_os = "linux"))]
use chen::{
    Position, SearchLimits,
    backend::{BatchEvaluator, CpuBatchEvaluator, SearchBackend},
    eval::{Material, Tapered},
    frontier::{FrontierConfig, FrontierSearch},
    hip::HipBatchEvaluator,
};

fn varied_positions(count: usize) -> Vec<Position> {
    let mut p = Position::startpos();
    let mut random = 0x1030_c4e5_u64;
    (0..count)
        .map(|_| {
            let moves = p.legal_moves();
            if moves.is_empty() || p.halfmove_clock() >= 100 {
                p = Position::startpos();
            } else {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                p.make_move(moves[random as usize % moves.len()]);
            }
            p
        })
        .collect()
}
#[test]
#[ignore = "requires real gfx1030 HIP runtime and CHEN_GPU_LIBRARY; run explicitly in ROCm container"]
fn actual_gfx1030_batch_parity_and_thread_transfer() {
    let positions = varied_positions(4096);
    for name in ["material", "tapered"] {
        let mut gpu =
            HipBatchEvaluator::from_environment(name, 4096).expect("real HIP backend required");
        assert!(gpu.device().architecture.starts_with("gfx1030"));
        assert!(gpu.device().total_memory > 0);
        let mut cpu = CpuBatchEvaluator::new(if name == "material" {
            Box::new(Material)
        } else {
            Box::new(Tapered)
        });
        for size in [0, 1, 7, 64, 257, 4096] {
            let mut actual = vec![i32::MIN; size];
            let mut expected = vec![0; size];
            let times = gpu.evaluate_batch(&positions[..size], &mut actual).unwrap();
            cpu.evaluate_batch(&positions[..size], &mut expected)
                .unwrap();
            assert_eq!(actual, expected, "{name}: batch={size}");
            assert_eq!(times.positions, size as u64);
            assert!(times.kernel_ms.is_finite() && times.kernel_ms >= 0.0);
        }
        assert!(gpu.evaluate_batch(&positions[..1], &mut []).is_err());
        assert!(
            gpu.evaluate_batch(&vec![positions[0]; 4097], &mut vec![0; 4097])
                .is_err()
        );
        let p = positions[0];
        std::thread::spawn(move || {
            let mut score = [0];
            gpu.evaluate_batch(&[p], &mut score).unwrap();
        })
        .join()
        .unwrap();
    }
}
#[test]
#[ignore = "requires real gfx1030 HIP runtime and CHEN_GPU_LIBRARY; run explicitly in ROCm container"]
fn actual_gpu_search_matches_cpu_frontier() {
    let config = FrontierConfig {
        batch_size: 257,
        max_positions: 50_000,
    };
    let mut gpu = FrontierSearch::new(
        Box::new(HipBatchEvaluator::from_environment("tapered", config.batch_size).unwrap()),
        config.clone(),
    )
    .unwrap();
    let mut cpu =
        FrontierSearch::new(Box::new(CpuBatchEvaluator::new(Box::new(Tapered))), config).unwrap();
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/2n5/8/3N4/8/4P3/4K3 w - - 0 1",
    ] {
        let mut p = Position::from_fen(fen).unwrap();
        let before = p;
        let limits = SearchLimits {
            depth: 3,
            nodes: None,
            time: None,
        };
        let a = gpu
            .search(&mut p, limits.clone(), &[], &mut |_| {})
            .unwrap();
        let b = cpu.search(&mut p, limits, &[], &mut |_| {}).unwrap();
        assert_eq!(
            (a.depth, a.score, a.best_move, a.nodes, a.pv),
            (b.depth, b.score, b.best_move, b.nodes, b.pv)
        );
        assert_eq!(p, before);
        assert!(gpu.batch_telemetry().positions > 0);
    }
    let mut p = Position::startpos();
    let limit = SearchLimits {
        depth: 1,
        nodes: Some(21),
        time: None,
    };
    let a = gpu.search(&mut p, limit.clone(), &[], &mut |_| {}).unwrap();
    let b = cpu.search(&mut p, limit, &[], &mut |_| {}).unwrap();
    assert_eq!(
        (a.depth, a.score, a.best_move, a.nodes),
        (b.depth, b.score, b.best_move, b.nodes)
    );
    assert_eq!(a.depth, 1);
    assert!(!a.stopped);
}
