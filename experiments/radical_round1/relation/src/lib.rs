//! Root lanes, shared opponent operators, and exact existential elimination.
//! No successor Position is constructed by the relational kernels.
use chen::core::{KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks};
use chen::{Color, Move, Piece, Position};
type Cell = Option<(Color, Piece)>;
#[derive(Clone, Copy, Debug)]
pub struct Change {
    pub sq: u8,
    pub cell: Cell,
}
#[derive(Clone, Debug)]
pub struct Action {
    pub mv: Move,
    pub changes: Vec<Change>,
    pub rights: u8,
    pub ep: Option<u8>,
    pub halfmove: u16,
}
#[derive(Clone)]
pub struct Board {
    cells: [Cell; 64],
    side: Color,
    rights: u8,
    ep: Option<u8>,
    halfmove: u16,
}
impl Board {
    pub fn from_position(p: &Position) -> Self {
        Self {
            cells: std::array::from_fn(|s| p.piece_at(s as u8)),
            side: p.side_to_move(),
            rights: p.castling_rights(),
            ep: p.en_passant_square(),
            halfmove: p.halfmove_clock(),
        }
    }
    pub fn after(&self, a: &Action) -> Self {
        let mut b = self.clone();
        for c in &a.changes {
            b.cells[c.sq as usize] = c.cell;
        }
        b.side = b.side.opposite();
        b.rights = a.rights;
        b.ep = a.ep;
        b.halfmove = a.halfmove;
        b
    }
    fn action(&self, mv: Move) -> Action {
        let (c, p) = self.cells[mv.from as usize].unwrap();
        let mut changes = vec![
            Change {
                sq: mv.from,
                cell: None,
            },
            Change {
                sq: mv.to,
                cell: Some((c, mv.promotion.unwrap_or(p))),
            },
        ];
        let capture = self.cells[mv.to as usize].is_some();
        if p == Piece::Pawn && self.ep == Some(mv.to) && !capture {
            changes.push(Change {
                sq: if c == Color::White {
                    mv.to - 8
                } else {
                    mv.to + 8
                },
                cell: None,
            });
        }
        if p == Piece::King && mv.from.abs_diff(mv.to) == 2 {
            let (from, to) = if mv.to > mv.from {
                (mv.from + 3, mv.from + 1)
            } else {
                (mv.from - 4, mv.from - 1)
            };
            changes.push(Change {
                sq: from,
                cell: None,
            });
            changes.push(Change {
                sq: to,
                cell: Some((c, Piece::Rook)),
            });
        }
        let mut rights = self.rights;
        for sq in [mv.from, mv.to] {
            rights &= match sq {
                0 => !2,
                7 => !1,
                56 => !8,
                63 => !4,
                _ => 255,
            };
        }
        if p == Piece::King {
            rights &= if c == Color::White { !3 } else { !12 };
        }
        Action {
            mv,
            changes,
            rights,
            ep: if p == Piece::Pawn && mv.from.abs_diff(mv.to) == 16 {
                Some((mv.from + mv.to) / 2)
            } else {
                None
            },
            halfmove: if p == Piece::Pawn || capture {
                0
            } else {
                self.halfmove.saturating_add(1)
            },
        }
    }
}
#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub templates: usize,
    pub tested_templates: usize,
    pub pseudo_pairs: usize,
    pub tested_pairs: usize,
    pub legal_pairs: usize,
    pub ray_steps: usize,
    pub words: usize,
    pub retained_bytes: usize,
    pub retired: usize,
    pub exhausted: usize,
}
impl Stats {
    pub fn add(&mut self, o: &Self) {
        self.templates += o.templates;
        self.tested_templates += o.tested_templates;
        self.pseudo_pairs += o.pseudo_pairs;
        self.tested_pairs += o.tested_pairs;
        self.legal_pairs += o.legal_pairs;
        self.ray_steps += o.ray_steps;
        self.words += o.words;
        self.retained_bytes += o.retained_bytes;
        self.retired += o.retired;
        self.exhausted += o.exhausted;
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    HasReply,
    Mate,
    Stalemate,
    Unknown,
}
#[derive(Debug)]
pub struct Classified {
    pub mv: Move,
    pub check: bool,
    pub verdict: Verdict,
    pub witness: Option<Move>,
}
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Stream,
    Full,
    Grouped,
}
struct Planes {
    pieces: [[u64; 64]; 12],
    occ: [u64; 64],
    rights: [u64; 4],
    ep: [u64; 64],
    valid: u64,
    side: Color,
    kings: [u8; 2],
}
#[derive(Clone, Copy)]
struct Operator {
    mv: Move,
    piece: Piece,
    between: u64,
    castle: bool,
}
fn bitmask(n: usize) -> u64 {
    if n == 64 { !0 } else { (1u64 << n) - 1 }
}
fn between(a: u8, b: u8) -> u64 {
    let df = (b as i8 % 8 - a as i8 % 8).signum();
    let dr = (b as i8 / 8 - a as i8 / 8).signum();
    let mut f = a as i8 % 8 + df;
    let mut r = a as i8 / 8 + dr;
    let mut out = 0;
    while f != b as i8 % 8 || r != b as i8 / 8 {
        out |= 1u64 << (r * 8 + f);
        f += df;
        r += dr;
    }
    out
}
fn operators(b: &Board) -> Vec<Operator> {
    let mut out = Vec::new();
    for from in 0..64u8 {
        let Some((c, p)) = b.cells[from as usize] else {
            continue;
        };
        if c != b.side {
            continue;
        }
        let mut targets = match p {
            Piece::Knight => KNIGHT_ATTACKS[from as usize],
            Piece::King => KING_ATTACKS[from as usize],
            Piece::Bishop => bishop_attacks(from, 0),
            Piece::Rook => rook_attacks(from, 0),
            Piece::Queen => bishop_attacks(from, 0) | rook_attacks(from, 0),
            Piece::Pawn => {
                let d = if c == Color::White { 8i16 } else { -8 };
                let to = from as i16 + d;
                let mut t = PAWN_ATTACKS[c.index()][from as usize];
                if (0..64).contains(&to) {
                    t |= 1 << to;
                    if from / 8 == if c == Color::White { 1 } else { 6 } {
                        t |= 1 << (to + d);
                    }
                }
                t
            }
        };
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let mid = if matches!(p, Piece::Bishop | Piece::Rook | Piece::Queen)
                || (p == Piece::Pawn && from.abs_diff(to) == 16)
            {
                between(from, to)
            } else {
                0
            };
            if p == Piece::Pawn && (to / 8 == 0 || to / 8 == 7) {
                for promotion in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                    out.push(Operator {
                        mv: Move::new(from, to, Some(promotion)),
                        piece: p,
                        between: mid,
                        castle: false,
                    });
                }
            } else {
                out.push(Operator {
                    mv: Move::new(from, to, None),
                    piece: p,
                    between: mid,
                    castle: false,
                });
            }
        }
        if p == Piece::King && from == if c == Color::White { 4 } else { 60 } {
            for to in [from - 2, from + 2] {
                out.push(Operator {
                    mv: Move::new(from, to, None),
                    piece: p,
                    between: if to < from {
                        (1 << (from - 1)) | (1 << (from - 2)) | (1 << (from - 3))
                    } else {
                        (1 << (from + 1)) | (1 << (from + 2))
                    },
                    castle: true,
                });
            }
        }
    }
    out
}
impl Planes {
    fn new(base: &Board, actions: &[Action]) -> Self {
        let n = actions.len();
        let valid = bitmask(n);
        let mut s = Self {
            pieces: [[0; 64]; 12],
            occ: [0; 64],
            rights: [0; 4],
            ep: [0; 64],
            valid,
            side: base.side.opposite(),
            kings: [0; 2],
        };
        for sq in 0..64 {
            if let Some((c, p)) = base.cells[sq] {
                s.pieces[c.index() * 6 + p.index()][sq] = valid;
                s.occ[sq] = valid;
                if p == Piece::King {
                    s.kings[c.index()] = sq as u8;
                }
            }
        }
        for (i, a) in actions.iter().enumerate() {
            let bit = 1 << i;
            for ch in &a.changes {
                let sq = ch.sq as usize;
                if let Some((c, p)) = base.cells[sq] {
                    s.pieces[c.index() * 6 + p.index()][sq] &= !bit;
                }
                s.occ[sq] &= !bit;
                if let Some((c, p)) = ch.cell {
                    s.pieces[c.index() * 6 + p.index()][sq] |= bit;
                    s.occ[sq] |= bit;
                }
            }
            for r in 0..4 {
                if a.rights & (1 << r) != 0 {
                    s.rights[r] |= bit;
                }
            }
            if let Some(ep) = a.ep {
                s.ep[ep as usize] |= bit;
            }
        }
        s
    }
    fn source(&self, c: Color, p: Piece, sq: u8) -> u64 {
        self.pieces[c.index() * 6 + p.index()][sq as usize]
    }
    fn own(&self, c: Color, sq: u8) -> u64 {
        (0..6).fold(0, |v, p| v | self.pieces[c.index() * 6 + p][sq as usize])
    }
    fn pseudo(&self, o: Operator) -> u64 {
        let m = o.mv;
        let c = self.side;
        let mut ok = self.source(c, o.piece, m.from)
            & !self.own(c, m.to)
            & !self.source(c.opposite(), Piece::King, m.to);
        let mut mid = o.between;
        while mid != 0 {
            let sq = mid.trailing_zeros() as usize;
            mid &= mid - 1;
            ok &= !self.occ[sq];
        }
        if o.piece == Piece::Pawn {
            if m.from % 8 == m.to % 8 {
                ok &= !self.occ[m.to as usize];
            } else {
                ok &= self.own(c.opposite(), m.to) | self.ep[m.to as usize];
            }
        }
        if o.castle {
            let right = c.index() * 2 + usize::from(m.to < m.from);
            ok &= self.rights[right];
            let rook = if m.to > m.from {
                m.from + 3
            } else {
                m.from - 4
            };
            ok &= self.source(c, Piece::Rook, rook);
        }
        ok & self.valid
    }
    // Exact attack operator after one shared reply, restricted to lane mask.
    // Opponent reply pieces only alter blockers; all hostile source planes belong
    // to the root mover. Captured source planes are removed before ray reduction.
    fn attacked(&self, target: u8, op: Option<Operator>, mask: u64, stats: &mut Stats) -> u64 {
        let hostile = self.side.opposite();
        let mut vacated = 0u64;
        let mut occupied = 0u64;
        let mut removed = 0u64;
        let mut ep_removed = 0u64;
        let mut ep_lanes = 0;
        if let Some(o) = op {
            vacated |= 1 << o.mv.from;
            occupied |= 1 << o.mv.to;
            removed |= 1 << o.mv.to;
            if o.piece == Piece::Pawn && o.mv.from % 8 != o.mv.to % 8 {
                ep_removed = 1
                    << if self.side == Color::White {
                        o.mv.to - 8
                    } else {
                        o.mv.to + 8
                    };
                ep_lanes = self.ep[o.mv.to as usize];
            }
            if o.castle {
                let (from, to) = if o.mv.to > o.mv.from {
                    (o.mv.from + 3, o.mv.from + 1)
                } else {
                    (o.mv.from - 4, o.mv.from - 1)
                };
                vacated |= 1 << from;
                occupied |= 1 << to;
            }
        }
        let source = |p: Piece, sq: u8| {
            let bit = 1u64 << sq;
            let keep = if removed & bit != 0 {
                0
            } else {
                if ep_removed & bit != 0 { !ep_lanes } else { !0 }
            };
            self.source(hostile, p, sq) & keep
        };
        let occ = |sq: u8| {
            let bit = 1u64 << sq;
            if occupied & bit != 0 {
                mask
            } else if vacated & bit != 0 {
                0
            } else {
                self.occ[sq as usize] & if ep_removed & bit != 0 { !ep_lanes } else { !0 }
            }
        };
        let mut attacked = 0;
        for (p, mut sources) in [
            (Piece::Knight, KNIGHT_ATTACKS[target as usize]),
            (Piece::King, KING_ATTACKS[target as usize]),
            (
                Piece::Pawn,
                PAWN_ATTACKS[self.side.index()][target as usize],
            ),
        ] {
            while sources != 0 {
                let sq = sources.trailing_zeros() as u8;
                sources &= sources - 1;
                attacked |= source(p, sq);
            }
        }
        attacked &= mask;
        if attacked == mask {
            return attacked;
        }
        for (df, dr, p) in [
            (1, 0, Piece::Rook),
            (-1, 0, Piece::Rook),
            (0, 1, Piece::Rook),
            (0, -1, Piece::Rook),
            (1, 1, Piece::Bishop),
            (-1, 1, Piece::Bishop),
            (1, -1, Piece::Bishop),
            (-1, -1, Piece::Bishop),
        ] {
            let mut f = target as i8 % 8 + df;
            let mut r = target as i8 / 8 + dr;
            let mut live = mask & !attacked;
            while (0..8).contains(&f) && (0..8).contains(&r) && live != 0 {
                let sq = (r * 8 + f) as u8;
                attacked |= live & (source(p, sq) | source(Piece::Queen, sq));
                live &= !occ(sq) & !attacked;
                f += df;
                r += dr;
                stats.ray_steps += 1;
            }
        }
        attacked & mask
    }
    fn legal(&self, o: Operator, mask: u64, stats: &mut Stats) -> u64 {
        let target = if o.piece == Piece::King {
            o.mv.to
        } else {
            self.kings[self.side.index()]
        };
        let mut ok = mask & !self.attacked(target, Some(o), mask, stats);
        if o.castle {
            ok &= !self.attacked(o.mv.from, None, ok, stats);
            let transit = Operator {
                mv: Move::new(o.mv.from, (o.mv.from + o.mv.to) / 2, None),
                piece: Piece::King,
                between: 0,
                castle: false,
            };
            ok &= !self.attacked(transit.mv.to, Some(transit), ok, stats);
        }
        ok
    }
}
/// The exhaustive mode retains all legal relation rows. Stream mode discards a
/// lane immediately after a legal witness. Budget counts tested operator words;
/// exhaustion returns Unknown, never promotes absence of evidence to mate.
pub fn classify(
    base: &Board,
    actions: &[Action],
    mode: Mode,
    budget: usize,
) -> (Vec<Classified>, Stats) {
    classify_observe(base, actions, mode, budget, &mut |_, _| {})
}
/// Inspector for compact complete relation rows; each bit indexes one root
/// action inside this word. Used for full legal-reply differential validation.
pub type RelationObserver<'a> = dyn FnMut(usize, &[(Move, u64)]) + 'a;
pub fn classify_observe(
    base: &Board,
    actions: &[Action],
    mode: Mode,
    budget: usize,
    observer: &mut RelationObserver<'_>,
) -> (Vec<Classified>, Stats) {
    let mut result = Vec::new();
    let mut stats = Stats::default();
    let mut enemy = base.clone();
    enemy.side = enemy.side.opposite();
    let ops = operators(&enemy);
    for (word, chunk) in actions.chunks(64).enumerate() {
        let planes = Planes::new(base, chunk);
        stats.words += 1;
        stats.templates += ops.len();
        let check = planes.attacked(
            planes.kings[planes.side.index()],
            None,
            planes.valid,
            &mut stats,
        );
        let mut remaining = planes.valid;
        let mut witnessed = 0;
        let mut witnesses = [None; 64];
        let mut rows = Vec::new();
        let mut done = true;
        let mut ordered: Vec<_> = if matches!(mode, Mode::Grouped) {
            ops.iter()
                .copied()
                .map(|op| (op, planes.pseudo(op)))
                .collect()
        } else {
            Vec::new()
        };
        if matches!(mode, Mode::Grouped) {
            ordered.sort_by_key(|(_, mask)| std::cmp::Reverse(mask.count_ones()));
        }
        for index in 0..ops.len() {
            let (op, pseudo) = if matches!(mode, Mode::Grouped) {
                ordered[index]
            } else {
                let op = ops[index];
                (op, planes.pseudo(op))
            };
            stats.pseudo_pairs += pseudo.count_ones() as usize;
            let mask = if matches!(mode, Mode::Full) {
                pseudo
            } else {
                pseudo & remaining
            };
            if mask == 0 {
                continue;
            }
            if stats.tested_templates >= budget {
                done = false;
                break;
            }
            stats.tested_templates += 1;
            stats.tested_pairs += mask.count_ones() as usize;
            let legal = planes.legal(op, mask, &mut stats);
            stats.legal_pairs += legal.count_ones() as usize;
            if matches!(mode, Mode::Full) {
                rows.push((op.mv, legal));
            }
            let mut fresh = legal & !witnessed;
            while fresh != 0 {
                let i = fresh.trailing_zeros() as usize;
                fresh &= fresh - 1;
                witnesses[i] = Some(op.mv);
            }
            witnessed |= legal;
            remaining &= !legal;
            if remaining == 0 && !matches!(mode, Mode::Full) {
                break;
            }
        }
        stats.retained_bytes += rows.capacity() * std::mem::size_of::<(Move, u64)>();
        observer(word * 64, &rows);
        std::hint::black_box(&rows);
        stats.retired += witnessed.count_ones() as usize;
        if done {
            stats.exhausted += remaining.count_ones() as usize;
        }
        for (i, a) in chunk.iter().enumerate() {
            let bit = 1 << i;
            let checking = check & bit != 0;
            result.push(Classified {
                mv: a.mv,
                check: checking,
                verdict: if witnessed & bit != 0 {
                    Verdict::HasReply
                } else if !done {
                    Verdict::Unknown
                } else if checking {
                    Verdict::Mate
                } else {
                    Verdict::Stalemate
                },
                witness: witnesses[i],
            });
        }
    }
    (result, stats)
}
/// Own exact generator, using a singleton identity lane and shared operators.
/// Needed by bounded nested certificates; no conventional generator is called.
pub fn legal_actions(base: &Board) -> Vec<Action> {
    let identity = Action {
        mv: Move::new(0, 0, None),
        changes: Vec::new(),
        rights: base.rights,
        ep: base.ep,
        halfmove: base.halfmove,
    };
    let mut opposite = base.clone();
    opposite.side = base.side.opposite();
    let planes = Planes::new(&opposite, &[identity]);
    let mut stats = Stats::default();
    operators(base)
        .into_iter()
        .filter(|&o| planes.legal(o, planes.pseudo(o), &mut stats) != 0)
        .map(|o| base.action(o.mv))
        .collect()
}
pub fn root_actions(p: &Position) -> (Board, Vec<Action>) {
    let b = Board::from_position(p);
    let moves = legal_actions(&b);
    (b, moves)
}
/// Exact bounded certificate: an action permits an immediate opponent mate.
/// This nests existential reply elimination; it is bounded game reasoning, not
/// a new minimax theorem. Only one Board per root action, never one per pair.
pub fn mate_threats(
    base: &Board,
    actions: &[Action],
    budget: usize,
) -> Vec<(Move, Option<Move>, bool)> {
    actions
        .iter()
        .map(|a| {
            let child = base.after(a);
            let replies = legal_actions(&child);
            let (labels, _) = classify(&child, &replies, Mode::Grouped, budget);
            let witness = labels
                .iter()
                .find(|x| x.verdict == Verdict::Mate)
                .map(|x| x.mv);
            let complete =
                witness.is_some() || labels.iter().all(|x| x.verdict != Verdict::Unknown);
            (a.mv, witness, complete)
        })
        .collect()
}

/// Filter checking candidates by one attack predicate per root-lane word.
pub fn checking_actions(base: &Board, actions: &[Action]) -> Vec<Action> {
    let mut out = Vec::new();
    for chunk in actions.chunks(64) {
        let planes = Planes::new(base, chunk);
        let mut stats = Stats::default();
        let mask = planes.attacked(
            planes.kings[planes.side.index()],
            None,
            planes.valid,
            &mut stats,
        );
        for (i, a) in chunk.iter().enumerate() {
            if mask & (1 << i) != 0 {
                out.push(a.clone());
            }
        }
    }
    out
}
/// Current-position terminal classification, independent of clocks/history.
/// Callers must prioritize this over draw-clock adjudication.
pub fn root_terminal(base: &Board) -> Verdict {
    if !legal_actions(base).is_empty() {
        return Verdict::HasReply;
    }
    let identity = Action {
        mv: Move::new(0, 0, None),
        changes: Vec::new(),
        rights: base.rights,
        ep: base.ep,
        halfmove: base.halfmove,
    };
    let mut opposite = base.clone();
    opposite.side = base.side.opposite();
    let planes = Planes::new(&opposite, &[identity]);
    if planes.attacked(
        planes.kings[base.side.index()],
        None,
        1,
        &mut Stats::default(),
    ) != 0
    {
        Verdict::Mate
    } else {
        Verdict::Stalemate
    }
}
/// Front-end ablation: conventional exact root actions, same relational reply
/// consumer. This mode deliberately retains orthodox root generation.
pub fn orthodox_root_actions(p: &Position) -> (Board, Vec<Action>) {
    let b = Board::from_position(p);
    let actions = p.legal_moves().into_iter().map(|m| b.action(m)).collect();
    (b, actions)
}
/// Front-end ablation using the previous full codec relation.
pub fn codec_root_actions(p: &Position) -> (Board, Vec<Action>) {
    let b = Board::from_position(p);
    let actions = chen::codec::CodecPosition::from_position(p)
        .relation()
        .legal_moves()
        .into_iter()
        .map(|m| b.action(m))
        .collect();
    (b, actions)
}
