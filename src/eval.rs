//! Evaluation owns its caches. Scores are centipawns from the side to move.
//! Search clamps static scores to its non-mate range.
//! Hooks run after make / after unmake, so incremental evaluators can mirror the board.
use crate::core::{Color, Move, Piece, Position};

pub trait Evaluator: Send {
    fn name(&self) -> &'static str;
    fn evaluate(&mut self, position: &Position) -> i32;
    fn reset(&mut self, _position: &Position) {}
    fn after_make(&mut self, _position: &Position, _mv: Move) {}
    fn after_unmake(&mut self, _position: &Position, _mv: Move) {}
}

pub const VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];
const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[derive(Default)]
pub struct Material;
impl Evaluator for Material {
    fn name(&self) -> &'static str {
        "material"
    }
    fn evaluate(&mut self, p: &Position) -> i32 {
        let mut score = 0;
        for (piece, value) in PIECES.into_iter().zip(VALUES) {
            score += value
                * (p.pieces(Color::White, piece).count_ones() as i32
                    - p.pieces(Color::Black, piece).count_ones() as i32);
        }
        if p.side_to_move() == Color::White {
            score
        } else {
            -score
        }
    }
}

// Store both phases together because each visited square contributes to both.
// Signed white-frame entries for each color avoid runtime geometry and sign work.
const TAPERED_SQUARES: [[[(i32, i32); 64]; 6]; 2] = tapered_squares();
const PHASE_WEIGHTS: [i32; 6] = [0, 1, 1, 2, 4, 0];

const fn tapered_squares() -> [[[(i32, i32); 64]; 6]; 2] {
    let mut table = [[[(0, 0); 64]; 6]; 2];
    let mut piece = 0;
    while piece < 6 {
        let mut square = 0;
        while square < 64 {
            let file = (square % 8) as i32;
            let rank = (square / 8) as i32;
            let center = 14 - (2 * file - 7).abs() - (2 * rank - 7).abs();
            let (mg, eg) = match piece {
                0 => (rank * 8 + center * 2, rank * 14 + center),
                1 => (center * 6, center * 4),
                2 => (center * 3 + rank * 2, center * 3),
                3 => (rank * 2 + if rank == 6 { 18 } else { 0 }, center),
                4 => (center, center * 2),
                _ => (
                    -center * 5 - rank * 4
                        + if rank == 0 && (file == 2 || file == 6) {
                            35
                        } else {
                            0
                        },
                    center * 6,
                ),
            };
            table[0][piece][square] = (VALUES[piece] + mg, VALUES[piece] + eg);
            table[1][piece][square ^ 56] = (-VALUES[piece] - mg, -VALUES[piece] - eg);
            square += 1;
        }
        piece += 1;
    }
    table
}

/// Per-piece contribution in the white frame, before bishop-pair bonuses and tapering.
#[inline]
pub(crate) fn piece_terms(color: Color, piece: Piece, square: u8) -> (i32, i32, i32) {
    let (mg, eg) = TAPERED_SQUARES[color as usize][piece as usize][square as usize];
    (mg, eg, PHASE_WEIGHTS[piece as usize])
}

/// Nonlinear pair bonus, signed in the white frame.
#[inline]
pub(crate) fn bishop_pair_terms(color: Color, count: u32) -> (i32, i32) {
    if count < 2 {
        (0, 0)
    } else if color == Color::White {
        (25, 40)
    } else {
        (-25, -40)
    }
}

/// Preserve the evaluator's promoted-material phase cap and truncation toward zero.
#[inline]
pub(crate) fn taper(mg: i32, eg: i32, phase: i32) -> i32 {
    let phase = phase.min(24);
    (mg * phase + eg * (24 - phase)) / 24
}

/// Transparent handcrafted baseline: material, geometry, bishop pair, tapered king placement.
/// Geometry is computed at compile time; scores match the original arithmetic evaluator.
#[derive(Default)]
pub struct Tapered;
impl Evaluator for Tapered {
    fn name(&self) -> &'static str {
        "tapered"
    }
    fn evaluate(&mut self, p: &Position) -> i32 {
        let mut mg = 0;
        let mut eg = 0;
        let mut phase = 0;
        for color in [Color::White, Color::Black] {
            for (idx, piece) in PIECES.into_iter().enumerate() {
                let mut bb = p.pieces(color, piece);
                phase += PHASE_WEIGHTS[idx] * bb.count_ones() as i32;
                while bb != 0 {
                    let square = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    let (m, e, _) = piece_terms(color, piece, square);
                    mg += m;
                    eg += e;
                }
            }
            let (pair_mg, pair_eg) =
                bishop_pair_terms(color, p.pieces(color, Piece::Bishop).count_ones());
            mg += pair_mg;
            eg += pair_eg;
        }
        let score = taper(mg, eg, phase);
        if p.side_to_move() == Color::White {
            score
        } else {
            -score
        }
    }
}

pub fn evaluator(name: &str) -> Result<Box<dyn Evaluator>, String> {
    match name {
        "material" => Ok(Box::new(Material)),
        "tapered" => Ok(Box::new(Tapered)),
        _ => Err(format!("unknown evaluator: {name}")),
    }
}
