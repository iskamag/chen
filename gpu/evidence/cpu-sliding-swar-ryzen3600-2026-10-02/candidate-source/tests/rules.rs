use chen::{
    Color, Move, Piece, Position,
    core::PIECES,
    rules::{CpuRules, RuleMove, RuleState, RulesBackend},
};
#[test]
fn full_rule_packet_roundtrip_and_field_offsets() {
    assert_eq!(std::mem::size_of::<RuleState>(), 128);
    assert_eq!(std::mem::offset_of!(RuleState, side), 96);
    assert_eq!(std::mem::offset_of!(RuleState, ep_square), 104);
    assert_eq!(std::mem::offset_of!(RuleState, hash), 120);
    assert_eq!(std::mem::size_of::<RuleMove>(), 4);
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 65535 4294967295",
    ] {
        let p = Position::from_fen(fen).unwrap();
        assert_eq!(RuleState::from(&p).to_position().unwrap(), p);
    }
    let state = RuleState::from(&Position::startpos());
    for broken in [
        RuleState { side: 2, ..state },
        RuleState {
            castling: 16,
            ..state
        },
        RuleState {
            ep_square: 64,
            ..state
        },
        RuleState {
            halfmove: 65536,
            ..state
        },
        RuleState {
            reserved: 1,
            ..state
        },
        RuleState {
            hash: state.hash ^ 1,
            ..state
        },
        RuleState {
            fullmove: 0,
            ..state
        },
    ] {
        assert!(broken.to_position().is_err());
    }
}
#[test]
fn import_rejects_overlaps_and_move_packets_validate() {
    let p = Position::startpos();
    let mut boards = [[0; 6]; 2];
    for c in [Color::White, Color::Black] {
        for piece in PIECES {
            boards[c.index()][piece.index()] = p.pieces(c, piece);
        }
    }
    boards[0][Piece::Queen.index()] |= 1 << 4;
    assert!(Position::from_bitboards(boards, Color::White, 15, None, 0, 1).is_err());
    for promo in [
        None,
        Some(Piece::Knight),
        Some(Piece::Bishop),
        Some(Piece::Rook),
        Some(Piece::Queen),
    ] {
        let mv = Move::new(48, 56, promo);
        assert_eq!(Move::try_from(RuleMove::try_from(mv).unwrap()).unwrap(), mv);
    }
    assert!(RuleMove::try_from(Move::new(0, 63, Some(Piece::King))).is_err());
    assert!(
        Move::try_from(RuleMove {
            from: 64,
            ..RuleMove::default()
        })
        .is_err()
    );
}
#[test]
fn rules_oracle_batches_special_moves_and_saturation() {
    let mut rules = CpuRules;
    let cases = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1", "e5d6"),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", "a7b8n"),
        ("4k3/8/8/8/8/8/8/4K3 b - - 65535 4294967295", "e8d8"),
    ];
    let positions: Vec<_> = cases
        .iter()
        .map(|(fen, _)| Position::from_fen(fen).unwrap())
        .collect();
    let states: Vec<_> = positions.iter().map(RuleState::from).collect();
    let moves: Vec<_> = cases
        .iter()
        .zip(&positions)
        .map(|((_, mv), p)| p.parse_uci_move(mv).unwrap())
        .collect();
    let batch = rules.legal_batch(&states).unwrap();
    let next = rules.make_batch(&states, &moves).unwrap();
    for (i, p) in positions.iter().enumerate() {
        assert_eq!(batch.moves[i], p.legal_moves());
        let mut expected = *p;
        expected.make_move(moves[i]);
        assert_eq!(next[i].to_position().unwrap(), expected);
    }
    assert!(rules.make_batch(&states, &moves[..1]).is_err());
    assert!(
        rules
            .make_batch(&states[..1], &[Move::new(12, 36, None)])
            .is_err()
    );
    assert!(
        rules
            .legal_batch(&[RuleState {
                hash: 0,
                ..states[0]
            }])
            .is_err()
    );
}
