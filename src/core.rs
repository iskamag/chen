//! Chess rules and reversible positions, independent of evaluation and search.
//!
//! Squares use little-endian rank/file numbering (`a1 = 0`, `h8 = 63`).
//! Moves carry only their chess meaning; special-move behavior is derived from
//! the position. A position is small and `Copy`, making isolated experiments
//! easy. The initial generator favors auditable legality over clever pin code.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Color {
    White,
    Black,
}
impl Color {
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}
impl Piece {
    pub const fn index(self) -> usize {
        self as usize
    }
}
pub const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<Piece>,
}
impl Move {
    pub const fn new(from: u8, to: u8, promotion: Option<Piece>) -> Self {
        Self {
            from,
            to,
            promotion,
        }
    }
}
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", square_name(self.from), square_name(self.to))?;
        if let Some(p) = self.promotion {
            write!(f, "{}", piece_char(p).to_ascii_lowercase())?;
        }
        Ok(())
    }
}

pub fn square_name(square: u8) -> String {
    if square >= 64 {
        return "??".into();
    }
    format!(
        "{}{}",
        (b'a' + square % 8) as char,
        (b'1' + square / 8) as char
    )
}
pub fn parse_square(name: &str) -> Result<u8, String> {
    let b = name.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return Err(format!("invalid square: {name}"));
    }
    Ok((b[1] - b'1') * 8 + b[0] - b'a')
}

const WHITE_KING: u8 = 1;
const WHITE_QUEEN: u8 = 2;
const BLACK_KING: u8 = 4;
const BLACK_QUEEN: u8 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Position {
    boards: [[u64; 6]; 2],
    side: Color,
    castling: u8,
    ep: Option<u8>,
    halfmove: u16,
    fullmove: u32,
    key: u64,
}

/// Restores every field exactly, including counters and the hash. The baseline
/// intentionally uses a whole-state undo; search owns storage for its undo stack.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    previous: Position,
}

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Position {
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("valid start position")
    }

    /// Validates FEN structure and the local invariants required by move making.
    /// This does not attempt to prove that a position is reachable from startpos.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let fields: Vec<_> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err("FEN must have exactly six fields".into());
        }
        let mut position = Self {
            boards: [[0; 6]; 2],
            side: Color::White,
            castling: 0,
            ep: None,
            halfmove: 0,
            fullmove: 1,
            key: 0,
        };
        let ranks: Vec<_> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN board must have eight ranks".into());
        }
        for (r, rank) in ranks.iter().enumerate() {
            let mut file = 0usize;
            for ch in rank.chars() {
                if ('1'..='8').contains(&ch) {
                    file += (ch as u8 - b'0') as usize;
                } else {
                    let piece = match ch.to_ascii_lowercase() {
                        'p' => Piece::Pawn,
                        'n' => Piece::Knight,
                        'b' => Piece::Bishop,
                        'r' => Piece::Rook,
                        'q' => Piece::Queen,
                        'k' => Piece::King,
                        _ => return Err(format!("invalid FEN piece: {ch}")),
                    };
                    if file >= 8 {
                        return Err("FEN rank has more than eight squares".into());
                    }
                    let color = if ch.is_ascii_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    position.boards[color.index()][piece.index()] |= 1u64 << ((7 - r) * 8 + file);
                    file += 1;
                }
                if file > 8 {
                    return Err("FEN rank has more than eight squares".into());
                }
            }
            if file != 8 {
                return Err("FEN rank must have eight squares".into());
            }
        }
        position.side = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("invalid FEN side to move".into()),
        };
        for color in [Color::White, Color::Black] {
            if position.pieces(color, Piece::King).count_ones() != 1 {
                return Err("FEN requires exactly one king per side".into());
            }
            if position.pieces(color, Piece::Pawn) & 0xff00_0000_0000_00ff != 0 {
                return Err("FEN pawns cannot occupy first or eighth rank".into());
            }
        }
        if KING_ATTACKS[position.king_square(Color::White) as usize]
            & position.pieces(Color::Black, Piece::King)
            != 0
        {
            return Err("FEN kings cannot be adjacent".into());
        }
        if fields[2] != "-" {
            if fields[2].is_empty() {
                return Err("invalid FEN castling rights".into());
            }
            for ch in fields[2].chars() {
                let (right, king, rook, color) = match ch {
                    'K' => (WHITE_KING, 4, 7, Color::White),
                    'Q' => (WHITE_QUEEN, 4, 0, Color::White),
                    'k' => (BLACK_KING, 60, 63, Color::Black),
                    'q' => (BLACK_QUEEN, 60, 56, Color::Black),
                    _ => return Err("invalid FEN castling rights".into()),
                };
                if position.castling & right != 0 {
                    return Err("duplicate FEN castling right".into());
                }
                if position.piece_at(king) != Some((color, Piece::King))
                    || position.piece_at(rook) != Some((color, Piece::Rook))
                {
                    return Err("FEN castling right requires a home king and rook".into());
                }
                position.castling |= right;
            }
        }
        if fields[3] != "-" {
            let sq = parse_square(fields[3])?;
            let (rank, pawn_sq, origin_sq) = if position.side == Color::White {
                (5, sq.checked_sub(8), sq.checked_add(8))
            } else {
                (2, sq.checked_add(8), sq.checked_sub(8))
            };
            if sq / 8 != rank
                || position.piece_at(sq).is_some()
                || pawn_sq.and_then(|s| position.piece_at(s))
                    != Some((position.side.opposite(), Piece::Pawn))
                || origin_sq.and_then(|s| position.piece_at(s)).is_some()
            {
                return Err("invalid FEN en passant target".into());
            }
            position.ep = Some(sq);
        }
        if !fields[4].bytes().all(|b| b.is_ascii_digit())
            || !fields[5].bytes().all(|b| b.is_ascii_digit())
        {
            return Err("FEN move counters must be unsigned integers".into());
        }
        position.halfmove = fields[4]
            .parse()
            .map_err(|_| "invalid FEN halfmove clock")?;
        position.fullmove = fields[5]
            .parse()
            .map_err(|_| "invalid FEN fullmove number")?;
        if position.fullmove == 0 {
            return Err("FEN fullmove number must be positive".into());
        }
        position.key = position.recompute_hash();
        Ok(position)
    }

    pub fn to_fen(&self) -> String {
        let mut board = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                if let Some((color, piece)) = self.piece_at(rank * 8 + file) {
                    if empty != 0 {
                        board.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    let ch = piece_char(piece);
                    board.push(if color == Color::White {
                        ch
                    } else {
                        ch.to_ascii_lowercase()
                    });
                } else {
                    empty += 1;
                }
            }
            if empty != 0 {
                board.push((b'0' + empty) as char);
            }
            if rank != 0 {
                board.push('/');
            }
        }
        let mut rights = String::new();
        for (bit, ch) in [
            (WHITE_KING, 'K'),
            (WHITE_QUEEN, 'Q'),
            (BLACK_KING, 'k'),
            (BLACK_QUEEN, 'q'),
        ] {
            if self.castling & bit != 0 {
                rights.push(ch);
            }
        }
        if rights.is_empty() {
            rights.push('-');
        }
        format!(
            "{board} {} {rights} {} {} {}",
            if self.side == Color::White { "w" } else { "b" },
            self.ep.map(square_name).unwrap_or_else(|| "-".into()),
            self.halfmove,
            self.fullmove
        )
    }

    pub const fn side_to_move(&self) -> Color {
        self.side
    }
    pub const fn pieces(&self, color: Color, piece: Piece) -> u64 {
        self.boards[color.index()][piece.index()]
    }
    pub fn occupancy(&self, color: Color) -> u64 {
        self.boards[color.index()].iter().fold(0, |a, b| a | b)
    }
    pub fn occupied(&self) -> u64 {
        self.occupancy(Color::White) | self.occupancy(Color::Black)
    }
    pub const fn hash(&self) -> u64 {
        self.key
    }
    pub const fn halfmove_clock(&self) -> u16 {
        self.halfmove
    }
    pub const fn fullmove_number(&self) -> u32 {
        self.fullmove
    }
    pub const fn en_passant_square(&self) -> Option<u8> {
        self.ep
    }
    pub fn piece_at(&self, square: u8) -> Option<(Color, Piece)> {
        if square >= 64 {
            return None;
        }
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                if self.pieces(color, piece) & (1u64 << square) != 0 {
                    return Some((color, piece));
                }
            }
        }
        None
    }
    pub fn king_square(&self, color: Color) -> u8 {
        self.pieces(color, Piece::King).trailing_zeros() as u8
    }

    pub fn attackers_to(&self, square: u8, color: Color) -> u64 {
        if square >= 64 {
            return 0;
        }
        (PAWN_ATTACKS[color.opposite().index()][square as usize] & self.pieces(color, Piece::Pawn))
            | (KNIGHT_ATTACKS[square as usize] & self.pieces(color, Piece::Knight))
            | (KING_ATTACKS[square as usize] & self.pieces(color, Piece::King))
            | (bishop_attacks(square, self.occupied())
                & (self.pieces(color, Piece::Bishop) | self.pieces(color, Piece::Queen)))
            | (rook_attacks(square, self.occupied())
                & (self.pieces(color, Piece::Rook) | self.pieces(color, Piece::Queen)))
    }
    pub fn is_in_check(&self, color: Color) -> bool {
        self.attackers_to(self.king_square(color), color.opposite()) != 0
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);
        let own = self.occupancy(self.side);
        let enemy =
            self.occupancy(self.side.opposite()) & !self.pieces(self.side.opposite(), Piece::King);
        let occupied = self.occupied();
        let allowed = !own & !self.pieces(self.side.opposite(), Piece::King);
        let mut pawns = self.pieces(self.side, Piece::Pawn);
        while pawns != 0 {
            let from = pop_square(&mut pawns);
            let to = if self.side == Color::White {
                from + 8
            } else {
                from - 8
            };
            if occupied & (1u64 << to) == 0 {
                push_pawn_moves(&mut moves, from, to);
                if from / 8 == if self.side == Color::White { 1 } else { 6 } {
                    let double = if self.side == Color::White {
                        from + 16
                    } else {
                        from - 16
                    };
                    if occupied & (1u64 << double) == 0 {
                        moves.push(Move::new(from, double, None));
                    }
                }
            }
            let mut captures = PAWN_ATTACKS[self.side.index()][from as usize] & enemy;
            if let Some(ep) = self.ep {
                captures |= PAWN_ATTACKS[self.side.index()][from as usize] & (1u64 << ep);
            }
            while captures != 0 {
                push_pawn_moves(&mut moves, from, pop_square(&mut captures));
            }
        }
        for piece in [
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            let mut pieces = self.pieces(self.side, piece);
            while pieces != 0 {
                let from = pop_square(&mut pieces);
                let mut targets = match piece {
                    Piece::Knight => KNIGHT_ATTACKS[from as usize],
                    Piece::Bishop => bishop_attacks(from, occupied),
                    Piece::Rook => rook_attacks(from, occupied),
                    Piece::Queen => bishop_attacks(from, occupied) | rook_attacks(from, occupied),
                    Piece::King => KING_ATTACKS[from as usize],
                    Piece::Pawn => unreachable!(),
                } & allowed;
                while targets != 0 {
                    moves.push(Move::new(from, pop_square(&mut targets), None));
                }
            }
        }
        if !self.is_in_check(self.side) {
            let (base, kingside, queenside) = if self.side == Color::White {
                (0, WHITE_KING, WHITE_QUEEN)
            } else {
                (56, BLACK_KING, BLACK_QUEEN)
            };
            if self.castling & kingside != 0
                && occupied & ((1u64 << (base + 5)) | (1u64 << (base + 6))) == 0
                && self.attackers_to(base + 5, self.side.opposite()) == 0
                && self.attackers_to(base + 6, self.side.opposite()) == 0
            {
                moves.push(Move::new(base + 4, base + 6, None));
            }
            if self.castling & queenside != 0
                && occupied & ((1u64 << (base + 1)) | (1u64 << (base + 2)) | (1u64 << (base + 3)))
                    == 0
                && self.attackers_to(base + 3, self.side.opposite()) == 0
                && self.attackers_to(base + 2, self.side.opposite()) == 0
            {
                moves.push(Move::new(base + 4, base + 2, None));
            }
        }
        let side = self.side;
        let mut scratch = *self;
        moves.retain(|&mv| {
            let undo = scratch.make_move(mv);
            let legal = !scratch.is_in_check(side);
            scratch.unmake_move(mv, undo);
            legal
        });
        moves
    }

    /// Applies a legal move returned by `legal_moves` or `parse_uci_move`.
    /// Callers must not pass arbitrary move coordinates into this hot path.
    pub fn make_move(&mut self, mv: Move) -> Undo {
        let undo = Undo { previous: *self };
        let (color, piece) = self
            .piece_at(mv.from)
            .expect("move source must contain a piece");
        debug_assert_eq!(color, self.side);
        self.key ^= CASTLING_KEYS[self.castling as usize];
        if let Some(file) = self.effective_ep_file() {
            self.key ^= EP_KEYS[file as usize];
        }
        let mut capture = self.piece_at(mv.to);
        if piece == Piece::Pawn && self.ep == Some(mv.to) && capture.is_none() {
            let capture_square = if color == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            };
            self.toggle(color.opposite(), Piece::Pawn, capture_square);
            capture = Some((color.opposite(), Piece::Pawn));
        } else if let Some((captured_color, captured_piece)) = capture {
            self.toggle(captured_color, captured_piece, mv.to);
        }
        self.toggle(color, piece, mv.from);
        self.toggle(color, mv.promotion.unwrap_or(piece), mv.to);
        if piece == Piece::King && mv.from.abs_diff(mv.to) == 2 {
            let (rook_from, rook_to) = if mv.to > mv.from {
                (mv.from + 3, mv.from + 1)
            } else {
                (mv.from - 4, mv.from - 1)
            };
            self.toggle(color, Piece::Rook, rook_from);
            self.toggle(color, Piece::Rook, rook_to);
        }
        if piece == Piece::King {
            self.castling &= if color == Color::White {
                !(WHITE_KING | WHITE_QUEEN)
            } else {
                !(BLACK_KING | BLACK_QUEEN)
            };
        }
        for square in [mv.from, mv.to] {
            self.castling &= match square {
                0 => !WHITE_QUEEN,
                7 => !WHITE_KING,
                56 => !BLACK_QUEEN,
                63 => !BLACK_KING,
                _ => 0xff,
            };
        }
        self.ep = if piece == Piece::Pawn && mv.from.abs_diff(mv.to) == 16 {
            Some((mv.from + mv.to) / 2)
        } else {
            None
        };
        self.halfmove = if piece == Piece::Pawn || capture.is_some() {
            0
        } else {
            self.halfmove.saturating_add(1)
        };
        if color == Color::Black {
            self.fullmove = self.fullmove.saturating_add(1);
        }
        self.side = color.opposite();
        self.key ^= SIDE_KEY ^ CASTLING_KEYS[self.castling as usize];
        if let Some(file) = self.effective_ep_file() {
            self.key ^= EP_KEYS[file as usize];
        }
        undo
    }
    pub fn unmake_move(&mut self, _mv: Move, undo: Undo) {
        *self = undo.previous;
    }
    fn toggle(&mut self, color: Color, piece: Piece, square: u8) {
        self.boards[color.index()][piece.index()] ^= 1u64 << square;
        self.key ^= PIECE_KEYS[color.index()][piece.index()][square as usize];
    }

    pub fn parse_uci_move(&self, text: &str) -> Result<Move, String> {
        let b = text.as_bytes();
        if (b.len() != 4 && b.len() != 5) || !text.is_ascii() {
            return Err(format!("invalid UCI move: {text}"));
        }
        let from = parse_square(&text[..2])?;
        let to = parse_square(&text[2..4])?;
        let promotion = if b.len() == 5 {
            Some(match b[4] {
                b'n' => Piece::Knight,
                b'b' => Piece::Bishop,
                b'r' => Piece::Rook,
                b'q' => Piece::Queen,
                _ => return Err("promotion must be n, b, r, or q".into()),
            })
        } else {
            None
        };
        let mv = Move::new(from, to, promotion);
        if self.legal_moves().contains(&mv) {
            Ok(mv)
        } else {
            Err(format!("illegal move: {text}"))
        }
    }

    /// True only for material that cannot produce any legal mating sequence.
    pub fn insufficient_material(&self) -> bool {
        for color in [Color::White, Color::Black] {
            for piece in [Piece::Pawn, Piece::Rook, Piece::Queen] {
                if self.pieces(color, piece) != 0 {
                    return false;
                }
            }
        }
        let knights =
            self.pieces(Color::White, Piece::Knight) | self.pieces(Color::Black, Piece::Knight);
        let bishops =
            self.pieces(Color::White, Piece::Bishop) | self.pieces(Color::Black, Piece::Bishop);
        if (knights | bishops).count_ones() <= 1 {
            return true;
        }
        if knights != 0 {
            return false;
        }
        const DARK: u64 = 0xaa55_aa55_aa55_aa55;
        bishops & DARK == 0 || bishops & !DARK == 0
    }

    /// En passant changes repetition identity only when at least one legal
    /// capture exists. This also handles pawns pinned along the captured rank.
    fn effective_ep_file(&self) -> Option<u8> {
        let target = self.ep?;
        let mut candidates = PAWN_ATTACKS[self.side.opposite().index()][target as usize]
            & self.pieces(self.side, Piece::Pawn);
        while candidates != 0 {
            let from = pop_square(&mut candidates);
            let captured = if self.side == Color::White {
                target - 8
            } else {
                target + 8
            };
            let mut scratch = *self;
            scratch.boards[self.side.index()][Piece::Pawn.index()] ^=
                (1u64 << from) | (1u64 << target);
            scratch.boards[self.side.opposite().index()][Piece::Pawn.index()] ^= 1u64 << captured;
            if !scratch.is_in_check(self.side) {
                return Some(target % 8);
            }
        }
        None
    }

    /// Independent recomputation is useful when validating experimental undo
    /// and hashing implementations against the baseline.
    pub fn recompute_hash(&self) -> u64 {
        let mut key = CASTLING_KEYS[self.castling as usize];
        if self.side == Color::Black {
            key ^= SIDE_KEY;
        }
        if let Some(file) = self.effective_ep_file() {
            key ^= EP_KEYS[file as usize];
        }
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                let mut board = self.pieces(color, piece);
                while board != 0 {
                    key ^=
                        PIECE_KEYS[color.index()][piece.index()][pop_square(&mut board) as usize];
                }
            }
        }
        key
    }
}

pub fn perft(position: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = position.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0;
    for mv in moves {
        let undo = position.make_move(mv);
        nodes += perft(position, depth - 1);
        position.unmake_move(mv, undo);
    }
    nodes
}

fn piece_char(piece: Piece) -> char {
    match piece {
        Piece::Pawn => 'P',
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
    }
}
fn pop_square(board: &mut u64) -> u8 {
    let sq = board.trailing_zeros() as u8;
    *board &= *board - 1;
    sq
}
fn push_pawn_moves(moves: &mut Vec<Move>, from: u8, to: u8) {
    if to / 8 == 0 || to / 8 == 7 {
        for promotion in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
            moves.push(Move::new(from, to, Some(promotion)));
        }
    } else {
        moves.push(Move::new(from, to, None));
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
const BISHOP_STEPS: [(i8, i8); 4] = [(1, 1), (-1, 1), (1, -1), (-1, -1)];
const ROOK_STEPS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const fn leaper_table(steps: &[(i8, i8)]) -> [u64; 64] {
    let mut table = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let mut i = 0;
        while i < steps.len() {
            let x = (sq % 8) as i8 + steps[i].0;
            let y = (sq / 8) as i8 + steps[i].1;
            if x >= 0 && x < 8 && y >= 0 && y < 8 {
                table[sq] |= 1u64 << (y * 8 + x);
            }
            i += 1;
        }
        sq += 1;
    }
    table
}
pub const KNIGHT_ATTACKS: [u64; 64] = leaper_table(&KNIGHT_STEPS);
pub const KING_ATTACKS: [u64; 64] = leaper_table(&KING_STEPS);
pub const PAWN_ATTACKS: [[u64; 64]; 2] = [
    leaper_table(&[(-1, 1), (1, 1)]),
    leaper_table(&[(-1, -1), (1, -1)]),
];

fn sliding_attacks(square: u8, occupied: u64, steps: &[(i8, i8)]) -> u64 {
    let mut attacks = 0;
    for &(dx, dy) in steps {
        let mut x = (square % 8) as i8 + dx;
        let mut y = (square / 8) as i8 + dy;
        while (0..8).contains(&x) && (0..8).contains(&y) {
            let bit = 1u64 << (y * 8 + x);
            attacks |= bit;
            if occupied & bit != 0 {
                break;
            }
            x += dx;
            y += dy;
        }
    }
    attacks
}
pub fn bishop_attacks(square: u8, occupied: u64) -> u64 {
    sliding_attacks(square, occupied, &BISHOP_STEPS)
}
pub fn rook_attacks(square: u8, occupied: u64) -> u64 {
    sliding_attacks(square, occupied, &ROOK_STEPS)
}

const fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
const fn piece_keys() -> [[[u64; 64]; 6]; 2] {
    let mut table = [[[0; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut s = 0;
            while s < 64 {
                table[c][p][s] = splitmix(0x4348_454e_0000_0000 + (c * 384 + p * 64 + s) as u64);
                s += 1;
            }
            p += 1;
        }
        c += 1;
    }
    table
}
const fn hash_keys<const N: usize>(offset: u64) -> [u64; N] {
    let mut table = [0; N];
    let mut i = 0;
    while i < N {
        table[i] = splitmix(0x4348_454e_0000_0000 + offset + i as u64);
        i += 1;
    }
    table
}
const PIECE_KEYS: [[[u64; 64]; 6]; 2] = piece_keys();
const CASTLING_KEYS: [u64; 16] = hash_keys(768);
const EP_KEYS: [u64; 8] = hash_keys(784);
const SIDE_KEY: u64 = splitmix(0x4348_454e_0000_0000 + 792);
