use chen::codec::{Boundary, CodecPosition};
use chen::{Color, Piece, Position};

const FIXTURES: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "8/8/8/r4pPK/8/8/8/7k w - f6 0 1",
    "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1",
    "4k3/P6r/8/8/8/8/7p/4K3 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
    "4r2k/8/8/8/1b6/8/8/4K3 w - - 0 1",
    "7k/8/8/Q1Q1Q1Q1/8/Q1Q1Q1Q1/8/K7 w - - 0 1",
];
fn sorted(mut moves: Vec<chen::Move>) -> Vec<String> {
    let mut names: Vec<_> = moves.drain(..).map(|mv| mv.to_string()).collect();
    names.sort();
    names
}
fn check(position: &Position, all_attacks: bool) {
    let codec = CodecPosition::from_position(position);
    for sq in 0..64 {
        assert_eq!(codec.piece_at(sq), position.piece_at(sq));
    }
    let relation = codec.relation();
    assert_eq!(
        sorted(relation.legal_moves()),
        sorted(position.legal_moves()),
        "{}",
        position.to_fen()
    );
    for color in [Color::White, Color::Black] {
        for sq in 0..64 {
            assert_eq!(
                relation.attacked_before(color, sq),
                position.attackers_to(sq, color) != 0
            );
        }
    }
    for (index, decision) in relation.decisions().iter().enumerate() {
        let mut child = *position;
        child.make_move(decision.mv);
        for sq in 0..64 {
            assert_eq!(
                decision.patch.piece_at_after(position, sq),
                child.piece_at(sq),
                "{} {} square {sq}",
                position.to_fen(),
                decision.mv
            );
            if all_attacks {
                for color in [Color::White, Color::Black] {
                    assert_eq!(
                        relation.attacked_after(index, color, sq),
                        child.attackers_to(sq, color) != 0,
                        "{} {} {color:?} square {sq}",
                        position.to_fen(),
                        decision.mv
                    );
                }
            }
        }
        assert_eq!(
            decision.gives_check,
            child.is_in_check(position.side_to_move().opposite())
        );
        // Castling also has an intermediate-state obligation, tested by legal set equality.
        if decision.moved != Piece::King || decision.mv.from.abs_diff(decision.mv.to) != 2 {
            assert_eq!(decision.legal, !child.is_in_check(position.side_to_move()));
        }
    }
}
#[test]
fn boundary_composition_is_an_exact_associative_algebra() {
    // Three possible symbols: empty, non-attacking blocker, attacking blocker.
    let leaf = [
        Boundary::default(),
        Boundary {
            occupied: 1,
            attacker: 0,
        },
        Boundary {
            occupied: 1,
            attacker: 1,
        },
    ];
    for a in leaf {
        for b in leaf {
            for c in leaf {
                assert_eq!(a.then(b).then(c), a.then(b.then(c)));
                let expected = [a, b, c]
                    .into_iter()
                    .find(|x| x.occupied != 0)
                    .unwrap_or_default()
                    .attacker;
                assert_eq!(a.then(b).then(c).attacker, expected);
            }
        }
    }
    // The same algebra operates independently on every bit/alternative.
    let a = Boundary {
        occupied: 0x5555,
        attacker: 0x1111,
    };
    let b = Boundary {
        occupied: 0xaaaa,
        attacker: 0x8888,
    };
    assert_eq!(a.then(b).attacker, 0x9999);
}
#[test]
fn exact_sparse_patches_and_all_counterfactual_attacks() {
    for fen in FIXTURES {
        check(&Position::from_fen(fen).unwrap(), true);
    }
    let wide = CodecPosition::from_position(&Position::from_fen(FIXTURES.last().unwrap()).unwrap())
        .relation();
    assert!(
        wide.stats().lane_words > 1,
        "exercise cross-word alternatives"
    );
}
#[test]
fn deterministic_legal_walks_match_orthodox_kernel() {
    let mut random = 0x8ac6_c0de_2026_1015u64;
    for fen in FIXTURES {
        let mut position = Position::from_fen(fen).unwrap();
        for ply in 0..96 {
            check(&position, ply % 24 == 0);
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            position.make_move(moves[random as usize % moves.len()]);
        }
    }
}
fn codec_perft(position: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    CodecPosition::from_position(position)
        .relation()
        .legal_moves()
        .into_iter()
        .map(|mv| {
            let mut child = *position;
            child.make_move(mv);
            codec_perft(&child, depth - 1)
        })
        .sum()
}
#[test]
fn independently_composed_legality_matches_known_perft() {
    assert_eq!(codec_perft(&Position::startpos(), 3), 8_902);
    assert_eq!(
        codec_perft(&Position::from_fen(FIXTURES[1]).unwrap(), 3),
        97_862
    );
    assert_eq!(
        codec_perft(&Position::from_fen(FIXTURES[2]).unwrap(), 3),
        2_812
    );
}

#[test]
fn line_scans_exclude_the_target_and_stop_at_the_nearest_blocker() {
    let position = Position::from_fen("7k/8/8/8/R2N3r/8/8/7K w - - 0 1").unwrap();
    let relation = CodecPosition::from_position(&position).relation();
    // The occupied d4 target is attacked from either end. Beyond it, the
    // blocker suppresses the more distant slider rather than the nearest one.
    for file in 0..8 {
        assert_eq!(
            relation.attacked_before(Color::White, 24 + file),
            (1..=3).contains(&file)
        );
        assert_eq!(
            relation.attacked_before(Color::Black, 24 + file),
            (3..=6).contains(&file)
        );
    }
    check(&position, true);
}

#[test]
fn boundary_work_is_linear_in_board_channels_and_alternative_words() {
    for fen in FIXTURES {
        let relation = CodecPosition::from_position(&Position::from_fen(fen).unwrap()).relation();
        // Four line families each visit 64 squares in two directions, for two
        // colors. More alternatives add words, never another per-target ray fold.
        assert_eq!(
            relation.stats().boundary_compositions,
            1024 * relation.stats().lane_words
        );
    }
}
