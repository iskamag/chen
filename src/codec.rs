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
    attacks: Vec<[u64; 128]>,
    legal_words: Vec<u64>,
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
    /// Candidate i is bit i. Parent, castling-transit and padding lanes are excluded.
    pub fn legal_words(&self) -> &[u64] {
        &self.legal_words
    }
    /// Attack predicates for whole alternative classes, in the same coordinates
    /// as `legal_words`. Consumers can intersect channels without unpacking boards.
    pub fn attacked_words(&self, by: Color, target: u8) -> impl Iterator<Item = u64> + '_ {
        assert!(target < 64, "target outside board");
        let row = by.index() * 64 + target as usize;
        (0..self.legal_words.len()).map(move |word| {
            // Physical lane zero is the unchanged parent. Shift across the word
            // boundary so public bit zero denotes decision zero in every word.
            let low = self.attacks[word][row] >> 1;
            let high = self.attacks.get(word + 1).map_or(0, |next| next[row] << 63);
            let valid = (self.decisions.len() - word * 64).min(64);
            (low | high) & (u64::MAX >> (64 - valid))
        })
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
        self.attacks[lane / 64][by.index() * 64 + target as usize] & (1 << (lane % 64)) != 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Line {
    squares: [u8; 8],
    len: u8,
}

struct Geometry {
    lines: [Line; 46],
    memberships: [[(u8, u8); 4]; 64],
}
static GEOMETRY: Geometry = board_geometry();

const fn board_geometry() -> Geometry {
    let mut geometry = Geometry {
        lines: [Line {
            squares: [0; 8],
            len: 0,
        }; 46],
        memberships: [[(0, 0); 4]; 64],
    };
    let mut square = 0u8;
    while square < 64 {
        let rank = square / 8;
        let file = square % 8;
        let ids = [
            rank,
            8 + file,
            16 + (7 + rank as i8 - file as i8) as u8,
            31 + rank + file,
        ];
        let mut family = 0;
        while family < 4 {
            let id = ids[family] as usize;
            let slot = geometry.lines[id].len;
            geometry.lines[id].squares[slot as usize] = square;
            geometry.lines[id].len += 1;
            geometry.memberships[square as usize][family] = (ids[family], slot);
            family += 1;
        }
        square += 1;
    }
    geometry
}

/// The board is a lossless 4-bit symbol stream; movement uses four overlapping
/// families of 8-bit occupancy channels. This compresses the *alternative set*,
/// not a claim that its redundant topology uses less memory than bitboards.
#[derive(Clone, Debug)]
pub struct CodecPosition {
    symbols: [u8; 32],
    occupied_lines: [u8; 46],
    side: Color,
    castling: u8,
    ep: Option<u8>,
    kings: [u8; 2],
}
impl CodecPosition {
    pub fn from_position(position: &Position) -> Self {
        let mut codec = Self {
            symbols: [0; 32],
            occupied_lines: [0; 46],
            side: position.side_to_move(),
            castling: position.castling_rights(),
            ep: position.en_passant_square(),
            kings: [
                position.king_square(Color::White),
                position.king_square(Color::Black),
            ],
        };
        for color in [Color::White, Color::Black] {
            for piece in crate::core::PIECES {
                let mut sources = position.pieces(color, piece);
                while sources != 0 {
                    let sq = sources.trailing_zeros() as u8;
                    sources &= sources - 1;
                    let symbol = 1 + color as u8 * 6 + piece as u8;
                    codec.symbols[sq as usize / 2] |= symbol << ((sq % 2) * 4);
                    for &(id, slot) in &GEOMETRY.memberships[sq as usize] {
                        codec.occupied_lines[id as usize] |= 1 << slot;
                    }
                }
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
        let mut planes = vec![[[0u64; 64]; 12]; words];
        let mut occupied = vec![[0u64; 64]; words];
        let mut possible_sources = [[0u64; 6]; 2];
        for sq in 0..64u8 {
            if let Some((color, piece)) = self.piece_at(sq) {
                possible_sources[color.index()][piece.index()] |= 1 << sq;
                for word in 0..words {
                    planes[word][color.index() * 6 + piece.index()][sq as usize] = !0;
                    occupied[word][sq as usize] = !0;
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
                    planes[word][color.index() * 6 + piece.index()][edit.square as usize] &= !bit;
                }
                let cell = &mut occupied[word][edit.square as usize];
                if let Some((color, piece)) = edit.piece {
                    possible_sources[color.index()][piece.index()] |= 1 << edit.square;
                    planes[word][color.index() * 6 + piece.index()][edit.square as usize] |= bit;
                    *cell |= bit;
                } else {
                    *cell &= !bit;
                }
            }
        }
        let mut attacks = vec![[0u64; 128]; words];
        let mut compositions = 0;
        for color in [Color::White, Color::Black] {
            for word in 0..words {
                let planes = &planes[word];
                let occupied = &occupied[word];
                let attack_word = &mut attacks[word];
                // Propagate only populated source planes. The alternatives stay
                // bit-sliced; a source mask reaches all destinations together.
                for (piece, table) in [
                    (Piece::Pawn, &crate::core::PAWN_ATTACKS[color.index()]),
                    (Piece::Knight, &crate::core::KNIGHT_ATTACKS),
                    (Piece::King, &crate::core::KING_ATTACKS),
                ] {
                    let mut sources = possible_sources[color.index()][piece.index()];
                    while sources != 0 {
                        let source = sources.trailing_zeros() as u8;
                        sources &= sources - 1;
                        let alternatives =
                            planes[color.index() * 6 + piece.index()][source as usize];
                        if alternatives == 0 {
                            continue;
                        }
                        let mut destinations = table[source as usize];
                        while destinations != 0 {
                            let target = destinations.trailing_zeros() as usize;
                            destinations &= destinations - 1;
                            attack_word[color.index() * 64 + target] |= alternatives;
                        }
                    }
                }
                for (line_id, line) in GEOMETRY.lines.iter().enumerate() {
                    let slider = if line_id < 16 {
                        Piece::Rook
                    } else {
                        Piece::Bishop
                    };
                    // Before visiting the target, the boundary describes all
                    // preceding squares, with the nearest one first. A new
                    // square is prepended because it becomes nearest at the
                    // following target. Include blockers only AFTER answering
                    // this target, so occupied squares can themselves be attacked.
                    for reverse in [false, true] {
                        let mut boundary = Boundary::default();
                        for cursor in 0..line.len as usize {
                            let slot = if reverse {
                                line.len as usize - 1 - cursor
                            } else {
                                cursor
                            };
                            let square = line.squares[slot];
                            attack_word[color.index() * 64 + square as usize] |= boundary.attacker;
                            let hostile = planes[color.index() * 6 + slider.index()]
                                [square as usize]
                                | planes[color.index() * 6 + Piece::Queen.index()][square as usize];
                            boundary = Boundary {
                                occupied: occupied[square as usize],
                                attacker: hostile,
                            }
                            .then(boundary);
                            compositions += 1;
                        }
                    }
                }
            }
        }
        let stats = RelationStats {
            candidates: decisions.len(),
            lanes,
            lane_words: words,
            patch_edits: edits,
            boundary_compositions: compositions,
            attack_bytes: attacks.len() * std::mem::size_of::<[u64; 128]>(),
        };
        let mut relation = MoveRelation {
            decisions: Vec::new(),
            attacks,
            legal_words: vec![0; decisions.len().div_ceil(64)],
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
        for (index, decision) in decisions.iter().enumerate() {
            if decision.legal {
                relation.legal_words[index / 64] |= 1 << (index % 64);
            }
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
                        let (line_id, slot) = GEOMETRY.memberships[from as usize][family];
                        let line = GEOMETRY.lines[line_id as usize];
                        let reach = LINE_REACH[slot as usize]
                            [self.occupied_lines[line_id as usize] as usize];
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
