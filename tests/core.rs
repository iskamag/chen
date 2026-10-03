use chen::core::{Color, Piece, Position, parse_square, perft};

#[test]
fn tactical_generation_and_terminal_probe_match_full_legal_reference() {
    let fixtures = [
        Position::startpos().to_fen(),
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1".into(),
        "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1".into(),
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1".into(),
        "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1".into(),
        "8/8/8/r4pPK/8/8/8/7k w - f6 0 1".into(),
        "4k3/P6r/8/8/8/8/7p/4K3 w - - 0 1".into(),
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".into(),
    ];
    let mut random = 0xc4e0_2026_1003_u64;
    for fen in fixtures {
        let mut p = Position::from_fen(&fen).unwrap();
        for _ in 0..128 {
            let before = p;
            let full = p.legal_moves();
            let reference: Vec<_> = full
                .iter()
                .copied()
                .filter(|mv| {
                    mv.promotion.is_some()
                        || p.piece_at(mv.to).is_some()
                        || (p.piece_at(mv.from).unwrap().1 == Piece::Pawn
                            && mv.from % 8 != mv.to % 8)
                })
                .collect();
            assert_eq!(p.legal_tactical_moves(), reference, "{}", p.to_fen());
            assert_eq!(p.has_legal_move(), !full.is_empty(), "{}", p.to_fen());
            assert_eq!(p, before);
            if full.is_empty() {
                break;
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            p.make_move(full[random as usize % full.len()]);
        }
    }
}

#[test]
fn initial_position_perft() {
    let mut position = Position::startpos();
    let original = position;
    for (depth, expected) in [(0, 1), (1, 20), (2, 400), (3, 8_902), (4, 197_281)] {
        assert_eq!(perft(&mut position, depth), expected, "depth {depth}");
        assert_eq!(position, original);
    }
}

#[test]
fn kiwipete_perft() {
    let mut position =
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .unwrap();
    for (depth, expected) in [(1, 48), (2, 2_039), (3, 97_862)] {
        assert_eq!(perft(&mut position, depth), expected, "depth {depth}");
    }
}

#[test]
fn reference_perft_positions_cover_checks_promotions_and_ep() {
    for (fen, counts) in [
        (
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            [14, 191, 2_812],
        ),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            [6, 264, 9_467],
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            [44, 1_486, 62_379],
        ),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            [46, 2_079, 89_890],
        ),
    ] {
        let mut position = Position::from_fen(fen).unwrap();
        for (i, expected) in counts.into_iter().enumerate() {
            assert_eq!(
                perft(&mut position, i as u32 + 1),
                expected,
                "{fen}, depth {}",
                i + 1
            );
        }
    }
}

#[test]
fn fen_roundtrip_and_validation() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
        "4k3/8/8/8/3P4/8/8/4K3 b - d3 0 2",
    ] {
        assert_eq!(Position::from_fen(fen).unwrap().to_fen(), fen);
    }
    for fen in [
        "",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/8/K6k w - - 0",
        "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - a3 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
        "4k3/8/8/8/8/8/8/4K3 w - - 65536 1",
        "4k3/8/8/8/8/8/8/4KP2 w - - 0 1",
        "4k3/8/8/8/8/8/4k3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
        "4k3/8/8/8/8/8/8/9K w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - é6 0 1",
    ] {
        assert!(Position::from_fen(fen).is_err(), "accepted {fen}");
    }
}

#[test]
fn en_passant_obeys_pins_and_can_remove_a_checking_pawn() {
    for fen in [
        "8/8/8/r4pPK/8/8/8/7k w - f6 0 1",
        "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ] {
        let position = Position::from_fen(fen).unwrap();
        let forbidden = if fen.starts_with('8') { "g5f6" } else { "e5d6" };
        assert!(position.parse_uci_move(forbidden).is_err());
        let no_ep =
            Position::from_fen(&fen.replace(if forbidden == "g5f6" { "f6" } else { "d6" }, "-"))
                .unwrap();
        assert_eq!(
            position.hash(),
            no_ep.hash(),
            "illegal ep must not change repetition identity"
        );
    }
    let mut position = Position::from_fen("7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1").unwrap();
    assert!(position.is_in_check(Color::White));
    let mv = position.parse_uci_move("e5d6").unwrap();
    let undo = position.make_move(mv);
    assert!(!position.is_in_check(Color::White));
    assert_eq!(position.piece_at(parse_square("d5").unwrap()), None);
    assert_eq!(position.hash(), position.recompute_hash());
    position.unmake_move(mv, undo);
    assert!(position.is_in_check(Color::White));
}

#[test]
fn castling_requires_safe_transit_and_updates_rook_and_rights() {
    let position = Position::from_fen("k7/8/8/8/8/8/6p1/4K2R w K - 0 1").unwrap();
    assert!(position.parse_uci_move("e1g1").is_err());
    let mut position = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    let original = position;
    for notation in ["e1g1", "e1c1"] {
        let mv = position.parse_uci_move(notation).unwrap();
        let undo = position.make_move(mv);
        let rook_sq = if notation == "e1g1" { "f1" } else { "d1" };
        assert_eq!(
            position.piece_at(parse_square(rook_sq).unwrap()),
            Some((Color::White, Piece::Rook))
        );
        assert!(position.to_fen().contains(" b kq - "));
        assert_eq!(position.hash(), position.recompute_hash());
        position.unmake_move(mv, undo);
        assert_eq!(position, original);
    }
    let mv = position.parse_uci_move("a1a8").unwrap();
    position.make_move(mv);
    assert!(position.to_fen().contains(" b Kk - "));
}

#[test]
fn promotions_include_underpromotions_and_restore_exactly() {
    let mut position = Position::from_fen("1r5k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
    let original = position;
    for target in ["a8", "b8"] {
        for (suffix, piece) in [
            ('q', Piece::Queen),
            ('r', Piece::Rook),
            ('b', Piece::Bishop),
            ('n', Piece::Knight),
        ] {
            let mv = position
                .parse_uci_move(&format!("a7{target}{suffix}"))
                .unwrap();
            let undo = position.make_move(mv);
            assert_eq!(
                position.piece_at(parse_square(target).unwrap()),
                Some((Color::White, piece))
            );
            assert_eq!(position.hash(), position.recompute_hash());
            position.unmake_move(mv, undo);
            assert_eq!(position, original);
        }
    }
    assert!(position.parse_uci_move("a7a8").is_err());
    assert!(position.parse_uci_move("a7a8k").is_err());
    assert!(position.parse_uci_move("é7a8").is_err());
}

#[test]
fn deterministic_playout_hashes_and_undo() {
    let mut position = Position::startpos();
    let original = position;
    let mut history = Vec::new();
    let mut random = 0x4348_454e_u64;
    for _ in 0..300 {
        assert_eq!(position.hash(), position.recompute_hash());
        assert_eq!(
            position.hash(),
            Position::from_fen(&position.to_fen()).unwrap().hash()
        );
        let moves = position.legal_moves();
        if moves.is_empty() {
            break;
        }
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let mv = moves[random as usize % moves.len()];
        let undo = position.make_move(mv);
        history.push((mv, undo));
    }
    while let Some((mv, undo)) = history.pop() {
        position.unmake_move(mv, undo);
        assert_eq!(position.hash(), position.recompute_hash());
    }
    assert_eq!(position, original);
}

#[test]
fn material_draws_are_conservative() {
    for fen in [
        "7k/8/8/8/8/8/8/K7 w - - 0 1",
        "7k/8/8/8/8/8/8/KN6 w - - 0 1",
        "7k/8/8/8/8/8/8/KB6 w - - 0 1",
        "5b1k/8/8/8/8/8/8/K1B5 w - - 0 1",
    ] {
        assert!(
            Position::from_fen(fen).unwrap().insufficient_material(),
            "{fen}"
        );
    }
    for fen in [
        "7k/8/8/8/8/8/8/KNN5 w - - 0 1",
        "7k/8/8/8/8/8/8/KNB5 w - - 0 1",
        "7k/8/8/8/8/8/8/KR6 w - - 0 1",
    ] {
        assert!(
            !Position::from_fen(fen).unwrap().insufficient_material(),
            "{fen}"
        );
    }
}
