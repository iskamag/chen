use chen::{
    Position, SearchLimits,
    backend::SearchBackend,
    catnip::{CatnipConfig, CatnipSearch},
    search::MATE,
};
use std::{sync::atomic::Ordering, time::Duration};

fn run(
    fen: &str,
    config: CatnipConfig,
    nodes: Option<u64>,
) -> (chen::SearchResult, chen::catnip::CatnipStats) {
    let mut position = Position::from_fen(fen).unwrap();
    let before = position;
    let mut search = CatnipSearch::new(config).unwrap();
    let result = search
        .search(
            &mut position,
            SearchLimits {
                depth: 6,
                nodes,
                time: None,
            },
            &[],
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(position, before);
    let mut replay = before;
    for &mv in &result.pv {
        assert!(replay.legal_moves().contains(&mv), "illegal PV move {mv}");
        replay.make_move(mv);
    }
    if let Some(mv) = result.best_move {
        assert_eq!(Some(&mv), result.pv.first());
    }
    (result, search.stats())
}
#[test]
fn deterministic_cat_school_and_pv() {
    let config = CatnipConfig {
        theta: 0.1,
        ..CatnipConfig::default()
    };
    let (a, sa) = run(&Position::startpos().to_fen(), config, Some(700));
    let (b, sb) = run(&Position::startpos().to_fen(), config, Some(700));
    assert_eq!(
        (a.best_move, a.score, &a.pv, a.nodes),
        (b.best_move, b.score, &b.pv, b.nodes)
    );
    assert_eq!(sa, sb);
    assert!(sa.peak_walkers <= config.walkers);
    assert!(a.nodes <= 700);
    assert_eq!(a.depth, 0, "rollouts are not complete minimax horizons");
}
#[test]
fn infinity_ignores_quiet_root_but_has_legal_fallback() {
    let (result, stats) = run(
        &Position::startpos().to_fen(),
        CatnipConfig {
            theta: f32::INFINITY,
            ..CatnipConfig::default()
        },
        None,
    );
    assert!(result.best_move.is_some());
    assert_eq!(result.pv.len(), 1);
    assert_eq!(stats.ignored_boring, 20);
    assert_eq!(stats.peak_walkers, 0);
    assert_eq!(stats.fail_meow, 1);
}
#[test]
fn terminals_precede_attention_and_draw_clock() {
    let config = CatnipConfig {
        theta: f32::INFINITY,
        ..CatnipConfig::default()
    };
    let (mate, _) = run("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", config, None);
    assert_eq!(mate.score, -MATE);
    assert_eq!(mate.best_move, None);
    let (stale, _) = run("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", config, None);
    assert_eq!(stale.score, 0);
    assert_eq!(stale.best_move, None);
    let (win, _) = run("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", config, None);
    assert_eq!(win.score, MATE - 1);
    let mut after = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    after.make_move(win.best_move.unwrap());
    assert!(after.legal_moves().is_empty());
}
#[test]
fn mate_in_one_preserves_previously_retained_walker_peak() {
    // Native move order visits quiet queen moves before reaching the mating move.
    // With theta=0 those earlier children retain walkers before the early return.
    let config = CatnipConfig {
        theta: 0.0,
        ..CatnipConfig::default()
    };
    let (result, stats) = run("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", config, None);
    assert_eq!(result.score, MATE - 1);
    assert!(result.nodes > 2, "at least one earlier child was visited");
    assert!(stats.peak_walkers > 0);
    assert!(stats.peak_walkers <= config.walkers);
}
#[test]
fn root_repetition_is_exact() {
    let mut position = Position::startpos();
    let mut history = Vec::new();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        history.push(position.hash());
        let mv = position.parse_uci_move(text).unwrap();
        position.make_move(mv);
    }
    let mut search = CatnipSearch::new(CatnipConfig::default()).unwrap();
    let result = search
        .search(
            &mut position,
            SearchLimits::default(),
            &history,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(result.score, 0);
    assert_eq!(result.nodes, 0);
    assert_eq!(search.stats().peak_walkers, 0);
}
#[test]
fn zero_budget_precancel_and_zero_time_return_fallback() {
    for mode in 0..4 {
        let mut position = Position::startpos();
        let mut search = CatnipSearch::new(CatnipConfig::default()).unwrap();
        search.stop_handle().store(mode == 1, Ordering::Relaxed);
        let result = search
            .search(
                &mut position,
                SearchLimits {
                    depth: if mode == 3 { 0 } else { 6 },
                    nodes: if mode == 0 { Some(0) } else { None },
                    time: if mode == 2 {
                        Some(Duration::ZERO)
                    } else {
                        None
                    },
                },
                &[],
                &mut |_| {},
            )
            .unwrap();
        assert!(result.stopped);
        assert_eq!(result.nodes, 0);
        assert!(position.legal_moves().contains(&result.best_move.unwrap()));
    }
    for budget in [1, 2, 7, 21, 90] {
        let (result, _) = run(
            &Position::startpos().to_fen(),
            CatnipConfig::default(),
            Some(budget),
        );
        assert!(result.nodes <= budget);
    }
}
#[test]
fn delta_really_suppresses_noticing() {
    let fen = Position::startpos().to_fen();
    let config = CatnipConfig {
        theta: 0.0,
        max_plies: 4,
        ..CatnipConfig::default()
    };
    let (_, alert) = run(
        &fen,
        CatnipConfig {
            delta_cp: 0,
            ..config
        },
        None,
    );
    let (_, distracted) = run(
        &fen,
        CatnipConfig {
            delta_cp: i32::MAX,
            ..config
        },
        None,
    );
    assert!(alert.noticed_drift > 0);
    assert_eq!(distracted.noticed_drift, 0);
    assert!(distracted.fail_meow > alert.fail_meow);
}
#[test]
fn invalid_configuration_is_rejected() {
    for config in [
        CatnipConfig {
            theta: f32::NAN,
            ..CatnipConfig::default()
        },
        CatnipConfig {
            theta: -1.0,
            ..CatnipConfig::default()
        },
        CatnipConfig {
            delta_cp: -1,
            ..CatnipConfig::default()
        },
        CatnipConfig {
            walkers: 0,
            ..CatnipConfig::default()
        },
        CatnipConfig {
            walkers: 33,
            ..CatnipConfig::default()
        },
        CatnipConfig {
            max_plies: 0,
            ..CatnipConfig::default()
        },
    ] {
        assert!(CatnipSearch::new(config).is_err());
    }
}
