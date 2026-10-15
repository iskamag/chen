use chen::{Color, Evaluator, Piece, Position, Tapered};
#[path = "../benches/support/tapered_reference.rs"]
mod reference;
use reference::ArithmeticTapered;

const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

fn compare(position: &Position) {
    assert_eq!(
        Tapered.evaluate(position),
        ArithmeticTapered.evaluate(position),
        "{}",
        position.to_fen()
    );
}

#[test]
fn tapered_lookup_matches_arithmetic_at_every_piece_square() {
    for color in [Color::White, Color::Black] {
        for (piece_index, piece) in PIECES.into_iter().enumerate() {
            for square in 0..64 {
                if piece == Piece::Pawn && (square / 8 == 0 || square / 8 == 7) {
                    continue;
                }
                let mut boards = [[0u64; 6]; 2];
                boards[color as usize][piece_index] = 1 << square;
                for king_color in [color, color.opposite()] {
                    if piece == Piece::King && king_color == color {
                        continue;
                    }
                    let king_square = [0, 7, 56, 63]
                        .into_iter()
                        .find(|&candidate| {
                            candidate != square
                                && boards[king_color.opposite() as usize][5] & (1 << candidate) == 0
                                && (boards[king_color.opposite() as usize][5] == 0 || {
                                    let other = boards[king_color.opposite() as usize][5]
                                        .trailing_zeros()
                                        as i32;
                                    (candidate / 8 - other / 8).abs() > 1
                                        || (candidate % 8 - other % 8).abs() > 1
                                })
                        })
                        .unwrap();
                    boards[king_color as usize][5] = 1 << king_square;
                }
                for side in [Color::White, Color::Black] {
                    compare(&Position::from_bitboards(boards, side, 0, None, 0, 1).unwrap());
                }
            }
        }
    }
}

#[test]
fn tapered_lookup_matches_arithmetic_with_promoted_material_and_phase_clamp() {
    let mut random = 0x2026_1015_4e56_414c_u64;
    for i in 0..4096 {
        let mut boards = [[0u64; 6]; 2];
        boards[0][5] = 1;
        boards[1][5] = 1 << 63;
        for square in 1..63 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            // Vary density to exercise the entire phase range and promoted material.
            if random % 64 >= i % 64 {
                continue;
            }
            let piece = (random >> 8) as usize % 5;
            if piece == 0 && (square / 8 == 0 || square / 8 == 7) {
                continue;
            }
            boards[(random >> 16) as usize % 2][piece] |= 1 << square;
        }
        for side in [Color::White, Color::Black] {
            compare(&Position::from_bitboards(boards, side, 0, None, 0, 1).unwrap());
        }
    }
}

#[test]
fn tapered_lookup_matches_arithmetic_through_make_and_unmake() {
    let mut random = 0xc4e0_2026_1015_u64;
    for (_, fen) in include_str!("../benches/positions.txt")
        .lines()
        .filter_map(|line| line.split_once('|'))
        .filter(|(name, _)| !name.starts_with('#'))
    {
        let mut position = Position::from_fen(fen.trim()).unwrap();
        let original = position;
        let mut stack = Vec::new();
        for _ in 0..512 {
            compare(&position);
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let mv = moves[random as usize % moves.len()];
            stack.push((mv, position.make_move(mv)));
        }
        while let Some((mv, undo)) = stack.pop() {
            position.unmake_move(mv, undo);
            compare(&position);
        }
        assert_eq!(position, original);
    }
}
