//! Exact counterfactual chess in overlapping channels and bit-sliced alternatives.
//!
//! One lane denotes one sparse edit packet, not a materialized successor board.
//! Line boundary transfers compose attacks for every lane at once. The resulting
//! relation retains all-square attack answers for decision procedures to consume.
//! This is an experimental CPU mechanism, independent of alpha-beta and evaluation.

use crate::{Color, Move, Piece, Position};

/// An ordered ray segment's boundary meaning, across 64 alternatives.
/// `occupied` says that the segment stops a ray; `attacker` says its first
/// occupied square is a slider of the requested color and channel type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Boundary {
    pub occupied: u64,
    pub attacker: u64,
}
impl Boundary {
    /// Compose a near segment with a far segment. This operation is associative;
    /// the empty segment is its identity. No board or individual lane is decoded.
    pub const fn then(self, far: Self) -> Self {
        Self {
            occupied: self.occupied | far.occupied,
            attacker: self.attacker | (!self.occupied & far.attacker),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edit {
    pub square: u8,
    pub piece: Option<(Color, Piece)>,
}
/// At most four changed squares: castling needs four, en passant three.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Patch {
    edits: [Edit; 4],
    len: u8,
}
impl Patch {
    pub fn edits(&self) -> &[Edit] {
        &self.edits[..self.len as usize]
    }
    pub fn piece_at_after(&self, position: &Position, square: u8) -> Option<(Color, Piece)> {
        self.edits()
            .iter()
            .find(|edit| edit.square == square)
            .map_or_else(|| position.piece_at(square), |edit| edit.piece)
    }
    fn set(&mut self, square: u8, piece: Option<(Color, Piece)>) {
        if let Some(edit) = self.edits[..self.len as usize]
            .iter_mut()
            .find(|edit| edit.square == square)
        {
            edit.piece = piece;
        } else {
            self.edits[self.len as usize] = Edit { square, piece };
            self.len += 1;
        }
    }
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub mv: Move,
    pub patch: Patch,
    pub moved: Piece,
    pub captured: Option<Piece>,
    pub legal: bool,
    pub gives_check: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RelationStats {
    pub candidates: usize,
    /// Includes the unchanged base and castling transit alternatives.
    pub lanes: usize,
    pub lane_words: usize,
    pub patch_edits: usize,
    pub boundary_compositions: usize,
    /// Retained all-square, two-color Boolean attack relation (not scratch).
    pub attack_bytes: usize,
}

/// Exact answers for all candidate moves. Candidate index is `decisions()` index.
/// Illegal candidates are retained because their refutations are useful evidence.
#[derive(Clone, Debug)]
pub struct MoveRelation {
    decisions: Vec<Decision>,
    attacks: Vec<u64>,
    stats: RelationStats,
}
impl MoveRelation {
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }
    pub fn legal_moves(&self) -> Vec<Move> {
        self.decisions
            .iter()
            .filter(|d| d.legal)
            .map(|d| d.mv)
            .collect()
    }
    pub fn stats(&self) -> RelationStats {
        self.stats
    }
    pub fn attacked_after(&self, candidate: usize, by: Color, target: u8) -> bool {
        assert!(
            candidate < self.decisions.len(),
            "candidate outside relation"
        );
        self.attacked_lane(candidate + 1, by, target)
    }
    pub fn attacked_before(&self, by: Color, target: u8) -> bool {
        self.attacked_lane(0, by, target)
    }
    fn attacked_lane(&self, lane: usize, by: Color, target: u8) -> bool {
        assert!(target < 64, "target outside board");
        self.attacks[(by.index() * 64 + target as usize) * self.stats.lane_words + lane / 64]
            & (1 << (lane % 64))
            != 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Line {
    squares: [u8; 8],
    len: u8,
    occupied: u8,
}

/// The board is a lossless 4-bit symbol stream; movement uses four overlapping
/// families of 8-bit occupancy channels. This compresses the *alternative set*,
/// not a claim that its redundant topology uses less memory than bitboards.
#[derive(Clone, Debug)]
pub struct CodecPosition {
    symbols: [u8; 32],
    lines: [Line; 46],
    memberships: [[(u8, u8); 4]; 64],
    side: Color,
    castling: u8,
    ep: Option<u8>,
    kings: [u8; 2],
}
impl CodecPosition {
    pub fn from_position(position: &Position) -> Self {
        let mut codec = Self {
            symbols: [0; 32],
            lines: [Line::default(); 46],
            memberships: [[(0, 0); 4]; 64],
            side: position.side_to_move(),
            castling: position.castling_rights(),
            ep: position.en_passant_square(),
            kings: [
                position.king_square(Color::White),
                position.king_square(Color::Black),
            ],
        };
        for sq in 0..64u8 {
            let symbol = position
                .piece_at(sq)
                .map_or(0, |(c, p)| 1 + c as u8 * 6 + p as u8);
            codec.symbols[sq as usize / 2] |= symbol << ((sq % 2) * 4);
            let rank = sq / 8;
            let file = sq % 8;
            let ids = [
                rank,
                8 + file,
                16 + (7 + rank as i8 - file as i8) as u8,
                31 + rank + file,
            ];
            for (family, id) in ids.into_iter().enumerate() {
                let line = &mut codec.lines[id as usize];
                let slot = line.len;
                line.squares[slot as usize] = sq;
                line.len += 1;
                if symbol != 0 {
                    line.occupied |= 1 << slot;
                }
                codec.memberships[sq as usize][family] = (id, slot);
            }
        }
        codec
    }
    pub fn piece_at(&self, square: u8) -> Option<(Color, Piece)> {
        if square >= 64 {
            return None;
        }
        let symbol = self.symbols[square as usize / 2] >> ((square % 2) * 4) & 15;
        if symbol == 0 {
            return None;
        }
        let color = if symbol <= 6 {
            Color::White
        } else {
            Color::Black
        };
        Some((color, crate::core::PIECES[(symbol as usize - 1) % 6]))
    }

    /// Compose the complete one-ply decision relation without making successors.
    pub fn relation(&self) -> MoveRelation {
        let mut decisions = self.candidates();
        let mut transits = Vec::new();
        // Auxiliary lanes preserve the rook's original position while the king
        // crosses the transit square. The final castling lane moves both pieces.
        for (index, decision) in decisions.iter().enumerate() {
            if decision.moved == Piece::King && decision.mv.from.abs_diff(decision.mv.to) == 2 {
                let transit = (decision.mv.from + decision.mv.to) / 2;
                let mut patch = Patch::default();
                patch.set(decision.mv.from, None);
                patch.set(transit, Some((self.side, Piece::King)));
                transits.push((index, transit, patch));
            }
        }
        let lanes = 1 + decisions.len() + transits.len();
        let words = lanes.div_ceil(64);
        let mut planes = vec![0u64; 12 * 64 * words];
        let mut occupied = vec![0u64; 64 * words];
        for sq in 0..64u8 {
            if let Some((color, piece)) = self.piece_at(sq) {
                for word in 0..words {
                    planes[plane_index(color, piece, sq, word, words)] = !0;
                    occupied[sq as usize * words + word] = !0;
                }
            }
        }
        let mut edits = 0;
        let patches = decisions
            .iter()
            .map(|d| &d.patch)
            .chain(transits.iter().map(|t| &t.2));
        for (index, patch) in patches.enumerate() {
            let lane = index + 1;
            let word = lane / 64;
            let bit = 1u64 << (lane % 64);
            for edit in patch.edits() {
                edits += 1;
                if let Some((color, piece)) = self.piece_at(edit.square) {
                    planes[plane_index(color, piece, edit.square, word, words)] &= !bit;
                }
                let cell = &mut occupied[edit.square as usize * words + word];
                if let Some((color, piece)) = edit.piece {
                    planes[plane_index(color, piece, edit.square, word, words)] |= bit;
                    *cell |= bit;
                } else {
                    *cell &= !bit;
                }
            }
        }
        let mut attacks = vec![0u64; 128 * words];
        let mut compositions = 0;
        for color in [Color::White, Color::Black] {
            for target in 0..64u8 {
                for word in 0..words {
                    let mut attack = 0;
                    for (piece, offsets) in [
                        (Piece::Knight, &KNIGHT_STEPS[..]),
                        (Piece::King, &KING_STEPS[..]),
                    ] {
                        for &(df, dr) in offsets {
                            if let Some(source) = offset(target, df, dr) {
                                attack |= planes[plane_index(color, piece, source, word, words)];
                            }
                        }
                    }
                    let pawn_back = if color == Color::White { -1 } else { 1 };
                    for df in [-1, 1] {
                        if let Some(source) = offset(target, df, pawn_back) {
                            attack |= planes[plane_index(color, Piece::Pawn, source, word, words)];
                        }
                    }
                    // Each channel's near-to-far fold is an exact finite-state
                    // boundary composition operating on 64 alternatives at once.
                    for (family, &(line_id, slot)) in
                        self.memberships[target as usize].iter().enumerate()
                    {
                        let line = self.lines[line_id as usize];
                        let slider = if family < 2 {
                            Piece::Rook
                        } else {
                            Piece::Bishop
                        };
                        for direction in [-1i8, 1] {
                            let mut boundary = Boundary::default();
                            let mut cursor = slot as i8 + direction;
                            while (0..line.len as i8).contains(&cursor) {
                                let sq = line.squares[cursor as usize];
                                let hostile = planes[plane_index(color, slider, sq, word, words)]
                                    | planes[plane_index(color, Piece::Queen, sq, word, words)];
                                boundary = boundary.then(Boundary {
                                    occupied: occupied[sq as usize * words + word],
                                    attacker: hostile,
                                });
                                compositions += 1;
                                cursor += direction;
                            }
                            attack |= boundary.attacker;
                        }
                    }
                    attacks[(color.index() * 64 + target as usize) * words + word] = attack;
                }
            }
        }
        let stats = RelationStats {
            candidates: decisions.len(),
            lanes,
            lane_words: words,
            patch_edits: edits,
            boundary_compositions: compositions,
            attack_bytes: attacks.len() * 8,
        };
        let mut relation = MoveRelation {
            decisions: Vec::new(),
            attacks,
            stats,
        };
        for (index, decision) in decisions.iter_mut().enumerate() {
            let king = if decision.moved == Piece::King {
                decision.mv.to
            } else {
                self.kings[self.side.index()]
            };
            decision.legal = !relation.attacked_lane(index + 1, self.side.opposite(), king);
            decision.gives_check = relation.attacked_lane(
                index + 1,
                self.side,
                self.kings[self.side.opposite().index()],
            );
        }
        for (transit_index, &(index, square, _)) in transits.iter().enumerate() {
            decisions[index].legal &=
                !relation.attacked_lane(0, self.side.opposite(), self.kings[self.side.index()])
                    && !relation.attacked_lane(
                        1 + decisions.len() + transit_index,
                        self.side.opposite(),
                        square,
                    );
        }
        relation.decisions = decisions;
        relation
    }

    fn candidates(&self) -> Vec<Decision> {
        let mut result = Vec::with_capacity(64);
        for from in 0..64u8 {
            let Some((color, piece)) = self.piece_at(from) else {
                continue;
            };
            if color != self.side {
                continue;
            }
            let mut targets = 0u64;
            match piece {
                Piece::Pawn => {
                    let dr = if color == Color::White { 1 } else { -1 };
                    if let Some(to) = offset(from, 0, dr)
                        && self.piece_at(to).is_none()
                    {
                        targets |= 1 << to;
                        let home_rank = if color == Color::White { 1 } else { 6 };
                        if from / 8 == home_rank
                            && let Some(double) = offset(from, 0, dr * 2)
                            && self.piece_at(double).is_none()
                        {
                            targets |= 1 << double;
                        }
                    }
                    for df in [-1, 1] {
                        if let Some(to) = offset(from, df, dr)
                            && (self.ep == Some(to)
                                || self
                                    .piece_at(to)
                                    .is_some_and(|(c, p)| c != color && p != Piece::King))
                        {
                            targets |= 1 << to;
                        }
                    }
                }
                Piece::Knight | Piece::King => {
                    let steps = if piece == Piece::Knight {
                        &KNIGHT_STEPS
                    } else {
                        &KING_STEPS
                    };
                    for &(df, dr) in steps {
                        if let Some(to) = offset(from, df, dr) {
                            targets |= 1 << to;
                        }
                    }
                }
                Piece::Bishop | Piece::Rook | Piece::Queen => {
                    for family in 0..4 {
                        if (piece == Piece::Bishop && family < 2)
                            || (piece == Piece::Rook && family >= 2)
                        {
                            continue;
                        }
                        let (line_id, slot) = self.memberships[from as usize][family];
                        let line = self.lines[line_id as usize];
                        let reach = LINE_REACH[slot as usize][line.occupied as usize];
                        for cursor in 0..line.len {
                            if reach & (1 << cursor) != 0 {
                                targets |= 1 << line.squares[cursor as usize];
                            }
                        }
                    }
                }
            }
            while targets != 0 {
                let to = targets.trailing_zeros() as u8;
                targets &= targets - 1;
                if self
                    .piece_at(to)
                    .is_some_and(|(c, p)| c == color || p == Piece::King)
                {
                    continue;
                }
                if piece == Piece::Pawn && (to / 8 == 0 || to / 8 == 7) {
                    for promotion in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                        result.push(self.decision(Move::new(from, to, Some(promotion)), piece));
                    }
                } else {
                    result.push(self.decision(Move::new(from, to, None), piece));
                }
            }
        }
        let base = if self.side == Color::White { 0 } else { 56 };
        let rights = self.castling >> if self.side == Color::White { 0 } else { 2 };
        for (right, empty, to) in [(1, &[5u8, 6][..], 6), (2, &[1u8, 2, 3][..], 2)] {
            if rights & right != 0 && empty.iter().all(|sq| self.piece_at(base + sq).is_none()) {
                result.push(self.decision(Move::new(base + 4, base + to, None), Piece::King));
            }
        }
        result
    }
    fn decision(&self, mv: Move, moved: Piece) -> Decision {
        let mut patch = Patch::default();
        patch.set(mv.from, None);
        patch.set(mv.to, Some((self.side, mv.promotion.unwrap_or(moved))));
        let mut captured = self.piece_at(mv.to).map(|(_, piece)| piece);
        if moved == Piece::Pawn && self.ep == Some(mv.to) && captured.is_none() {
            let captured_square = if self.side == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            };
            patch.set(captured_square, None);
            captured = Some(Piece::Pawn);
        }
        if moved == Piece::King && mv.from.abs_diff(mv.to) == 2 {
            let (from, to) = if mv.to > mv.from {
                (mv.from + 3, mv.from + 1)
            } else {
                (mv.from - 4, mv.from - 1)
            };
            patch.set(from, None);
            patch.set(to, Some((self.side, Piece::Rook)));
        }
        Decision {
            mv,
            patch,
            moved,
            captured,
            legal: false,
            gives_check: false,
        }
    }
}
fn plane_index(color: Color, piece: Piece, sq: u8, word: usize, words: usize) -> usize {
    ((color.index() * 6 + piece.index()) * 64 + sq as usize) * words + word
}
const KNIGHT_STEPS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_STEPS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
fn offset(sq: u8, df: i8, dr: i8) -> Option<u8> {
    let f = (sq % 8) as i8 + df;
    let r = (sq / 8) as i8 + dr;
    ((0..8).contains(&f) && (0..8).contains(&r)).then_some((r * 8 + f) as u8)
}
// A complete, cache-small finite line table (8 slots x 256 occupancy symbols).
const LINE_REACH: [[u8; 256]; 8] = line_reach();
const fn line_reach() -> [[u8; 256]; 8] {
    let mut result = [[0; 256]; 8];
    let mut slot = 0;
    while slot < 8 {
        let mut occupied = 0;
        while occupied < 256 {
            let mut reach = 0;
            let mut cursor = slot + 1;
            while cursor < 8 {
                reach |= 1 << cursor;
                if occupied & (1 << cursor) != 0 {
                    break;
                }
                cursor += 1;
            }
            let mut cursor = slot;
            while cursor > 0 {
                cursor -= 1;
                reach |= 1 << cursor;
                if occupied & (1 << cursor) != 0 {
                    break;
                }
            }
            result[slot][occupied] = reach;
            occupied += 1;
        }
        slot += 1;
    }
    result
}
