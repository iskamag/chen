use chen::{
    Color, Evaluator, Position, SearchLimits, Tapered,
    backend::SearchBackend,
    codec::CodecPosition,
    codec_policy::{CodecPolicy, Preferences},
    search::MATE,
};
use std::{sync::atomic::Ordering, time::Duration};

fn check_patch_scores(position: &Position) -> usize {
    let relation = CodecPosition::from_position(position).relation();
    let preferences = Preferences::new(position, &relation);
    let mut count = 0;
    for (index, decision) in relation.decisions().iter().enumerate() {
        let preference = preferences.score(index);
        assert_eq!(preference.is_some(), decision.legal);
        let Some(preference) = preference else {
            continue;
        };
        let mut child = *position;
        child.make_move(decision.mv);
        assert_eq!(
            preference.static_cp,
            -Tapered.evaluate(&child),
            "{} / {}",
            position.to_fen(),
            decision.mv
        );
        assert_eq!(preference.mv, decision.mv);
        assert_eq!(
            preference.gives_check,
            child.is_in_check(child.side_to_move())
        );
        count += 1;
    }
    count
}

#[test]
fn sparse_scores_equal_full_tapered_evaluation_including_special_moves() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1",
        "4k3/8/8/8/8/8/1b6/2B1KB2 w - - 0 1",
        "4k3/qqqqqqqq/8/8/8/8/QQQQQQQQ/4K3 w - - 0 1",
    ];
    let mut checked = 0;
    for fen in fens {
        checked += check_patch_scores(&Position::from_fen(fen).unwrap());
    }
    let mut position = Position::startpos();
    let mut seed = 0x6a706567_u64;
    for _ in 0..256 {
        checked += check_patch_scores(&position);
        let moves = position.legal_moves();
        if moves.is_empty() {
            position = Position::startpos();
            continue;
        }
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        position.make_move(moves[(seed >> 32) as usize % moves.len()]);
    }
    assert!(checked > 1000);
}

#[test]
fn policy_selects_from_relation_without_claiming_search_depth() {
    let mut policy = CodecPolicy::default();
    let mut position = Position::from_fen("4k3/8/8/8/8/8/3q4/3QK3 w - - 0 1").unwrap();
    let before = position;
    let mut reports = 0;
    let result = policy
        .search(&mut position, SearchLimits::default(), &[], &mut |_| {
            reports += 1
        })
        .unwrap();
    assert_eq!(position, before);
    assert_eq!(reports, 1);
    assert_eq!(result.best_move.unwrap().to, 11);
    assert_eq!(
        position.piece_at(11),
        Some((Color::Black, chen::Piece::Queen))
    );
    assert_eq!(result.depth, 0);
    assert_eq!((result.nodes, result.qnodes), (0, 0));
    assert_eq!(result.pv, vec![result.best_move.unwrap()]);
    let stats = policy.stats();
    assert_eq!(stats.scored_alternatives, stats.legal_alternatives);
    assert!(stats.legal_alternatives > 0);
    assert!(!result.stopped);
}

#[test]
fn root_mate_and_draws_precede_preferences() {
    let mut policy = CodecPolicy::default();
    for (fen, expected) in [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", -MATE),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 100 1", 0),
    ] {
        let mut position = Position::from_fen(fen).unwrap();
        let result = policy
            .search(&mut position, SearchLimits::default(), &[], &mut |_| {})
            .unwrap();
        assert_eq!(result.score, expected, "{fen}");
        assert_eq!(policy.stats().scored_alternatives, 0);
    }
    let mut position = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 8 1").unwrap();
    let hash = position.hash();
    let result = policy
        .search(
            &mut position,
            SearchLimits::default(),
            &[hash, 1, hash, 2],
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(result.score, 0);
    assert_eq!(policy.stats().scored_alternatives, 0);
}

#[test]
fn interrupted_relation_keeps_legal_fallback_and_can_be_reused() {
    let mut policy = CodecPolicy::default();
    let stop = policy.stop_handle();
    stop.store(true, Ordering::Relaxed);
    let mut position = Position::startpos();
    let before = position;
    let result = policy
        .search(&mut position, SearchLimits::default(), &[], &mut |_| {})
        .unwrap();
    assert!(result.stopped);
    assert_eq!(policy.stats().scored_alternatives, 0);
    assert!(position.legal_moves().contains(&result.best_move.unwrap()));
    stop.store(false, Ordering::Relaxed);
    let result = policy
        .search(
            &mut position,
            SearchLimits {
                time: Some(Duration::ZERO),
                ..SearchLimits::default()
            },
            &[],
            &mut |_| {},
        )
        .unwrap();
    assert!(result.stopped);
    assert_eq!(position, before);
    let result = policy
        .search(&mut position, SearchLimits::default(), &[], &mut |_| {})
        .unwrap();
    assert!(!result.stopped);
    assert_eq!(position.side_to_move(), Color::White);
    assert_eq!(policy.stats().scored_alternatives, 20);
}

#[test]
fn optional_book_respects_disable_and_root_draw_authority() {
    use chen::OpeningBook;
    use std::sync::Arc;
    let book = Arc::new(OpeningBook::from_pgn_tsv("A\tfixture\t1. e4 e5\n").unwrap());
    let mut policy = CodecPolicy::default();
    let mut position = Position::startpos();
    policy.set_book(Some(book), 0, 12);
    let result = policy
        .search(&mut position, SearchLimits::default(), &[], &mut |_| {})
        .unwrap();
    assert_eq!(result.best_move.unwrap().to_string(), "e2e4");
    assert!(result.book_move.is_some());
    assert_eq!(policy.stats().score_kind, "book");
    assert_eq!(policy.stats().scored_alternatives, 0);
    let mut drawn =
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1").unwrap();
    let result = policy
        .search(&mut drawn, SearchLimits::default(), &[], &mut |_| {})
        .unwrap();
    assert_eq!(result.score, 0);
    assert!(result.book_move.is_none());
    assert_eq!(policy.stats().score_kind, "root_terminal");
    policy.set_book(None, 0, 12);
    let result = policy
        .search(&mut position, SearchLimits::default(), &[], &mut |_| {})
        .unwrap();
    assert!(result.book_move.is_none());
    assert_eq!(policy.stats().score_kind, "static_heuristic");
}
