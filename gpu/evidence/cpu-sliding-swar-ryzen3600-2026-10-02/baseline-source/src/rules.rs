//! Stable full-rule state and interchangeable batched rule implementations.
//! Device packets carry exact clocks/hash; repetition remains separate path state.
use crate::{Color, Move, Piece, Position, core::PIECES};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleState {
    pub pieces: [u64; 12],
    pub side: u32,
    pub castling: u32,
    pub ep_square: i32,
    pub halfmove: u32,
    pub fullmove: u32,
    pub reserved: u32,
    pub hash: u64,
}
impl From<&Position> for RuleState {
    fn from(p: &Position) -> Self {
        let mut pieces = [0; 12];
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                pieces[color.index() * 6 + piece.index()] = p.pieces(color, piece);
            }
        }
        Self {
            pieces,
            side: p.side_to_move().index() as u32,
            castling: p.castling_rights() as u32,
            ep_square: p.en_passant_square().map_or(-1, i32::from),
            halfmove: p.halfmove_clock() as u32,
            fullmove: p.fullmove_number(),
            reserved: 0,
            hash: p.hash(),
        }
    }
}
impl RuleState {
    pub fn to_position(&self) -> Result<Position, String> {
        if self.side > 1
            || self.castling > 15
            || !(-1..64).contains(&self.ep_square)
            || self.halfmove > u16::MAX as u32
            || self.reserved != 0
        {
            return Err("invalid rule-state metadata".into());
        }
        let boards = [
            self.pieces[..6].try_into().unwrap(),
            self.pieces[6..].try_into().unwrap(),
        ];
        let p = Position::from_bitboards(
            boards,
            if self.side == 0 {
                Color::White
            } else {
                Color::Black
            },
            self.castling as u8,
            (self.ep_square >= 0).then_some(self.ep_square as u8),
            self.halfmove as u16,
            self.fullmove,
        )?;
        if p.hash() != self.hash {
            return Err("rule-state hash mismatch".into());
        }
        Ok(p)
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuleMove {
    pub from: u8,
    pub to: u8,
    pub promotion: u8,
    pub reserved: u8,
}
impl TryFrom<Move> for RuleMove {
    type Error = String;
    fn try_from(mv: Move) -> Result<Self, String> {
        if mv.from >= 64 || mv.to >= 64 {
            return Err("invalid move coordinates".into());
        }
        let promotion = match mv.promotion {
            None => 0,
            Some(Piece::Knight) => 1,
            Some(Piece::Bishop) => 2,
            Some(Piece::Rook) => 3,
            Some(Piece::Queen) => 4,
            _ => return Err("invalid promotion piece".into()),
        };
        Ok(Self {
            from: mv.from,
            to: mv.to,
            promotion,
            reserved: 0,
        })
    }
}
impl TryFrom<RuleMove> for Move {
    type Error = String;
    fn try_from(mv: RuleMove) -> Result<Self, String> {
        if mv.from >= 64 || mv.to >= 64 || mv.reserved != 0 {
            return Err("invalid move packet".into());
        }
        let promotion = match mv.promotion {
            0 => None,
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            _ => return Err("invalid promotion packet".into()),
        };
        Ok(Self::new(mv.from, mv.to, promotion))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalBatch {
    pub moves: Vec<Vec<Move>>,
    pub in_check: Vec<bool>,
    pub insufficient: Vec<bool>,
}
/// External states/moves are validated; failed batches are discarded atomically.
pub trait RulesBackend: Send {
    fn name(&self) -> &str;
    fn legal_batch(&mut self, states: &[RuleState]) -> Result<LegalBatch, String>;
    fn make_batch(
        &mut self,
        states: &[RuleState],
        moves: &[Move],
    ) -> Result<Vec<RuleState>, String>;
}
#[derive(Default)]
pub struct CpuRules;
impl RulesBackend for CpuRules {
    fn name(&self) -> &str {
        "cpu-rules"
    }
    fn legal_batch(&mut self, states: &[RuleState]) -> Result<LegalBatch, String> {
        let positions: Vec<_> = states
            .iter()
            .map(RuleState::to_position)
            .collect::<Result<_, _>>()?;
        Ok(LegalBatch {
            moves: positions.iter().map(Position::legal_moves).collect(),
            in_check: positions
                .iter()
                .map(|p| p.is_in_check(p.side_to_move()))
                .collect(),
            insufficient: positions
                .iter()
                .map(Position::insufficient_material)
                .collect(),
        })
    }
    fn make_batch(
        &mut self,
        states: &[RuleState],
        moves: &[Move],
    ) -> Result<Vec<RuleState>, String> {
        if states.len() != moves.len() {
            return Err("rule batch length mismatch".into());
        }
        states
            .iter()
            .zip(moves)
            .map(|(state, &mv)| {
                let mut p = state.to_position()?;
                if !p.legal_moves().contains(&mv) {
                    return Err(format!("illegal transition: {mv}"));
                }
                p.make_move(mv);
                Ok(RuleState::from(&p))
            })
            .collect()
    }
}
