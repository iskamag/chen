//! End-to-end checks of the bundled opening book. Requires the build-time fetch.
use chen::{OpeningBook, Position};
use std::collections::HashSet;

#[test]
fn bundled_dataset_replays_completely() {
    let book = OpeningBook::bundled().expect("build.rs fetched the opening dataset");
    assert!(book.lines() > 3_000, "lines {}", book.lines());
    assert!(book.positions() > 5_000, "positions {}", book.positions());

    // The dataset strongly favors 1. e4 and 1. d4, so seeded choice must offer both.
    let start = Position::startpos();
    let mut seen = HashSet::new();
    for seed in 0..128 {
        let entry = book.probe(&start, 0, seed).expect("start is in book");
        assert!(start.legal_moves().contains(&entry.mv));
        assert!(!entry.name.is_empty());
        seen.insert(entry.uci());
    }
    assert!(seen.contains("e2e4") && seen.contains("d2d4"), "{seen:?}");
}

#[test]
fn probe_walks_a_legal_line_until_the_book_ends() {
    let book = OpeningBook::bundled().unwrap();
    let mut position = Position::startpos();
    let mut plies = 0;
    while let Some(entry) = book.probe(&position, plies, 12345) {
        assert!(
            position.legal_moves().contains(&entry.mv),
            "illegal book move at ply {plies}"
        );
        position.make_move(entry.mv);
        plies += 1;
        assert!(plies < 64, "book must run out before the game ends");
    }
    assert!(
        plies >= 2,
        "expected a multi-move opening line, got {plies}"
    );
}

#[test]
fn seeds_vary_the_first_move_across_games() {
    let book = OpeningBook::bundled().unwrap();
    let start = Position::startpos();
    let moves: HashSet<_> = (1..=64)
        .map(|seed| book.probe(&start, 0, seed).unwrap().uci())
        .collect();
    assert!(moves.len() >= 3, "expected variety, got {moves:?}");
}
