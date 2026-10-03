//! Canonical SAN for legal moves, following PGN sections 8.2.3.3–8.2.3.5.
//! https://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm

use chen::{Move, Piece, Position};

/// Format a legal move without changing the supplied position.
pub(crate) fn san(position: &Position, mv: Move) -> String {
    let (_, piece) = position.piece_at(mv.from).expect("legal move has a piece");
    let mut notation = String::with_capacity(8);
    if piece == Piece::King && mv.from.abs_diff(mv.to) == 2 {
        notation.push_str(if mv.to > mv.from { "O-O" } else { "O-O-O" });
    } else {
        // A legal diagonal pawn move into an empty square is en passant.
        let capture = position.piece_at(mv.to).is_some()
            || (piece == Piece::Pawn && mv.from % 8 != mv.to % 8);
        if piece == Piece::Pawn {
            if capture {
                notation.push(file(mv.from));
            }
        } else {
            notation.push(piece_letter(piece));
            let mut ambiguous = false;
            let mut same_file = false;
            let mut same_rank = false;
            // Only legal alternatives count: an absolutely pinned piece must
            // not force disambiguation even when it attacks the destination.
            for alternative in position.legal_moves() {
                if alternative.from != mv.from
                    && alternative.to == mv.to
                    && position.piece_at(alternative.from) == Some((position.side_to_move(), piece))
                {
                    ambiguous = true;
                    same_file |= alternative.from % 8 == mv.from % 8;
                    same_rank |= alternative.from / 8 == mv.from / 8;
                }
            }
            if ambiguous {
                if !same_file {
                    notation.push(file(mv.from));
                } else if !same_rank {
                    notation.push(rank(mv.from));
                } else {
                    notation.push(file(mv.from));
                    notation.push(rank(mv.from));
                }
            }
        }
        if capture {
            notation.push('x');
        }
        notation.push(file(mv.to));
        notation.push(rank(mv.to));
        if let Some(promotion) = mv.promotion {
            notation.push('=');
            notation.push(piece_letter(promotion));
        }
    }

    let mut after = *position;
    after.make_move(mv);
    if after.is_in_check(after.side_to_move()) {
        notation.push(if after.has_legal_move() { '+' } else { '#' });
    }
    notation
}

fn file(square: u8) -> char {
    (b'a' + square % 8) as char
}
fn rank(square: u8) -> char {
    (b'1' + square / 8) as char
}
fn piece_letter(piece: Piece) -> char {
    match piece {
        Piece::Pawn => 'P',
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legal(position: &Position, uci: &str) -> Move {
        position
            .legal_moves()
            .into_iter()
            .find(|mv| mv.to_string() == uci)
            .unwrap_or_else(|| panic!("{uci} must be legal in {}", position.to_fen()))
    }
    fn assert_san(fen: &str, uci: &str, expected: &str) {
        let position = Position::from_fen(fen).unwrap();
        let before = position;
        assert_eq!(san(&position, legal(&position, uci)), expected);
        assert_eq!(position, before, "SAN must not mutate root position");
    }

    #[test]
    fn ordinary_moves_and_captures() {
        let mut position = Position::startpos();
        for (uci, expected) in [
            ("e2e4", "e4"),
            ("d7d5", "d5"),
            ("e4d5", "exd5"),
            ("d8d5", "Qxd5"),
            ("b1c3", "Nc3"),
            ("d5d8", "Qd8"),
            ("f1c4", "Bc4"),
        ] {
            let mv = legal(&position, uci);
            assert_eq!(san(&position, mv), expected);
            position.make_move(mv);
        }
        assert_san("7k/8/8/8/8/8/8/4K3 w - - 0 1", "e1d2", "Kd2");
    }

    #[test]
    fn castling_both_sides_both_colors() {
        for (side, from, kingside, queenside) in [("w", "e1", "g1", "c1"), ("b", "e8", "g8", "c8")]
        {
            let fen = format!("r3k2r/8/8/8/8/8/8/R3K2R {side} KQkq - 0 1");
            assert_san(&fen, &format!("{from}{kingside}"), "O-O");
            assert_san(&fen, &format!("{from}{queenside}"), "O-O-O");
        }
        assert_san("5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", "O-O+");
    }

    #[test]
    fn en_passant_uses_normal_pawn_capture_notation() {
        assert_san("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", "exd6");
        assert_san("7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1", "e4d3", "exd3");
        assert_san("8/8/8/R2pP2k/8/8/8/K7 w - d6 0 1", "e5d6", "exd6+");
    }

    #[test]
    fn promotions_and_capture_promotions() {
        let fen = "7k/P7/8/8/8/8/8/7K w - - 0 1";
        for (uci, expected) in [
            ("a7a8q", "a8=Q+"),
            ("a7a8r", "a8=R+"),
            ("a7a8b", "a8=B"),
            ("a7a8n", "a8=N"),
        ] {
            assert_san(fen, uci, expected);
        }
        assert_san("k6r/6P1/8/8/8/8/8/K7 w - - 0 1", "g7h8q", "gxh8=Q+");
        assert_san("k7/8/8/8/8/8/6p1/K6R b - - 0 1", "g2h1q", "gxh1=Q+");
    }

    #[test]
    fn disambiguation_prefers_file_then_rank_then_full_square() {
        assert_san("K6k/8/8/8/8/8/8/1N3N2 w - - 0 1", "b1d2", "Nbd2");
        assert_san("K6k/8/8/8/8/1N6/8/1N6 w - - 0 1", "b1d2", "N1d2");
        assert_san("K6k/8/8/8/8/1N6/8/1N3N2 w - - 0 1", "b1d2", "Nb1d2");
        assert_san("K6k/8/8/8/8/8/3p4/1N3N2 w - - 0 1", "b1d2", "Nbxd2");
    }

    #[test]
    fn pinned_alternative_does_not_force_disambiguation() {
        assert_san("4r2k/8/8/8/8/8/R3R3/4K3 w - - 0 1", "a2d2", "Rd2");
    }

    #[test]
    fn checkmate_and_stalemate_suffixes() {
        let mut position = Position::startpos();
        for uci in ["f2f3", "e7e5", "g2g4"] {
            let mv = legal(&position, uci);
            position.make_move(mv);
        }
        assert_eq!(san(&position, legal(&position, "d8h4")), "Qh4#");
        // A terminal draw does not receive a check or mate marker.
        let mut position = Position::from_fen("7k/5K2/8/5Q2/8/8/8/8 w - - 0 1").unwrap();
        let mv = legal(&position, "f5g6");
        assert_eq!(san(&position, mv), "Qg6");
        position.make_move(mv);
        assert!(!position.is_in_check(position.side_to_move()));
        assert!(!position.has_legal_move());
    }
}
