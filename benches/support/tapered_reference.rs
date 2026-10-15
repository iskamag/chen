//! Frozen arithmetic evaluator from c4c9096, for differential tests and profiling.
use chen::core::{Color, Piece, Position};
use chen::eval::{Evaluator, VALUES};
const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];
/// Transparent handcrafted baseline: material, geometry, bishop pair, tapered king placement.
#[derive(Default)]
pub struct ArithmeticTapered;
impl Evaluator for ArithmeticTapered {
    fn name(&self) -> &'static str {
        "tapered"
    }
    fn evaluate(&mut self, p: &Position) -> i32 {
        let mut mg = 0;
        let mut eg = 0;
        let mut phase = 0;
        for color in [Color::White, Color::Black] {
            let sign = if color == Color::White { 1 } else { -1 };
            for (idx, piece) in PIECES.into_iter().enumerate() {
                let mut bb = p.pieces(color, piece);
                phase += [0, 1, 1, 2, 4, 0][idx] * bb.count_ones() as i32;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as i32;
                    bb &= bb - 1;
                    let file = sq % 8;
                    let rank = if color == Color::White {
                        sq / 8
                    } else {
                        7 - sq / 8
                    };
                    let center = 14 - (2 * file - 7).abs() - (2 * rank - 7).abs();
                    let (m, e) = match piece {
                        Piece::Pawn => (rank * 8 + center * 2, rank * 14 + center),
                        Piece::Knight => (center * 6, center * 4),
                        Piece::Bishop => (center * 3 + rank * 2, center * 3),
                        Piece::Rook => (rank * 2 + if rank == 6 { 18 } else { 0 }, center),
                        Piece::Queen => (center, center * 2),
                        Piece::King => (
                            -center * 5 - rank * 4
                                + if rank == 0 && (file == 2 || file == 6) {
                                    35
                                } else {
                                    0
                                },
                            center * 6,
                        ),
                    };
                    mg += sign * (VALUES[idx] + m);
                    eg += sign * (VALUES[idx] + e);
                }
            }
            if p.pieces(color, Piece::Bishop).count_ones() >= 2 {
                mg += sign * 25;
                eg += sign * 40;
            }
        }
        let phase = phase.min(24);
        let score = (mg * phase + eg * (24 - phase)) / 24;
        if p.side_to_move() == Color::White {
            score
        } else {
            -score
        }
    }
}
