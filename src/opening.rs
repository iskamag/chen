//! Opening book built from the CC0 Lichess `chess-openings` dataset.
//!
//! `build.rs` fetches the pinned SAN lines into `OUT_DIR`; `bundled` replays
//! them through Chen's own rules and rejects any line that does not parse, so a
//! corrupt or changed dataset degrades to "no book" instead of wrong moves.
//! The book is a position graph: identical positions reached by different move
//! orders share one node, so transpositions are covered without extra entries.

use crate::core::{Move, Piece, Position, parse_square};
use std::{collections::HashMap, fmt};

/// Moves are chosen from the weighted candidate edges. A seed of zero asks for
/// the most common move; any other seed mixes in `position.hash()` so one book
/// can vary per game while staying reproducible.
#[derive(Clone, Debug)]
pub struct BookMove {
    pub mv: Move,
    /// Opening name of the position the move leads into, when the dataset names it.
    pub name: String,
}
impl BookMove {
    pub fn uci(&self) -> String {
        self.mv.to_string()
    }
}

#[derive(Default)]
struct Node {
    /// Shortest authored line ending exactly here; empty for unnamed positions.
    name: String,
    name_plies: usize,
    total: u32,
    moves: Vec<(Move, u32)>,
}

#[derive(Default)]
pub struct OpeningBook {
    nodes: HashMap<u64, Node>,
    lines: usize,
    max_plies: usize,
}

impl fmt::Debug for OpeningBook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpeningBook")
            .field("lines", &self.lines)
            .field("positions", &self.nodes.len())
            .field("max_plies", &self.max_plies)
            .finish()
    }
}

impl OpeningBook {
    /// Replay the five bundled ECO volumes, or `None` if the build had no network.
    pub fn bundled() -> Option<Self> {
        let text = [
            include_str!(concat!(env!("OUT_DIR"), "/chen-openings/a.tsv")),
            include_str!(concat!(env!("OUT_DIR"), "/chen-openings/b.tsv")),
            include_str!(concat!(env!("OUT_DIR"), "/chen-openings/c.tsv")),
            include_str!(concat!(env!("OUT_DIR"), "/chen-openings/d.tsv")),
            include_str!(concat!(env!("OUT_DIR"), "/chen-openings/e.tsv")),
        ]
        .join("\n");
        Self::from_pgn_tsv(&text).ok().filter(|book| book.lines > 0)
    }

    /// Parse the upstream `eco\tname\tpgn` table (SAN move lines).
    pub fn from_pgn_tsv(text: &str) -> Result<Self, String> {
        let mut book = Self::default();
        for (index, line) in text.lines().enumerate() {
            let mut fields = line.splitn(3, '\t');
            let (Some(_eco), Some(name), Some(pgn)) = (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            if name == "name" && pgn == "pgn" {
                continue; // header row
            }
            let moves = replay(pgn).map_err(|error| format!("line {}: {error}", index + 1))?;
            if !moves.is_empty() {
                book.insert_line(&moves, name);
            }
        }
        Ok(book)
    }

    pub fn lines(&self) -> usize {
        self.lines
    }
    pub fn positions(&self) -> usize {
        self.nodes.len()
    }

    fn insert_line(&mut self, moves: &[Move], name: &str) {
        self.lines += 1;
        self.max_plies = self.max_plies.max(moves.len());
        let mut position = Position::startpos();
        for (index, &mv) in moves.iter().enumerate() {
            let node = self.nodes.entry(position.hash()).or_default();
            node.total += 1;
            match node.moves.iter_mut().find(|(edge, _)| *edge == mv) {
                Some((_, weight)) => *weight += 1,
                None => node.moves.push((mv, 1)),
            }
            position.make_move(mv);
            if index + 1 == moves.len() {
                let child = self.nodes.entry(position.hash()).or_default();
                if child.name.is_empty() || moves.len() < child.name_plies {
                    child.name = name.to_owned();
                    child.name_plies = moves.len();
                }
            }
        }
    }

    /// Choose a move for `position` after `plies` half-moves, or `None` when the
    /// position is unnamed, the book is exhausted, or the ply cap is reached.
    pub fn probe(&self, position: &Position, plies: usize, seed: u64) -> Option<BookMove> {
        if plies >= self.max_plies {
            return None;
        }
        let key = position.hash();
        let node = self.nodes.get(&key)?;
        let total = u64::from(node.total);
        let draw = mix(seed ^ key) % total.max(1);
        let mut cumulative = 0;
        for &(mv, weight) in &node.moves {
            cumulative += u64::from(weight);
            if draw < cumulative {
                return Some(BookMove {
                    mv,
                    name: self
                        .name_for(position, mv)
                        .unwrap_or_else(|| node.name.clone()),
                });
            }
        }
        None
    }

    fn name_for(&self, position: &Position, mv: Move) -> Option<String> {
        let mut child = *position;
        child.make_move(mv);
        self.nodes
            .get(&child.hash())
            .map(|node| node.name.clone())
            .filter(|name| !name.is_empty())
    }
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Replay a PGN move-text field (e.g. `1. e4 e5 2. Nf3`) into legal moves.
fn replay(pgn: &str) -> Result<Vec<Move>, String> {
    let mut position = Position::startpos();
    let mut moves = Vec::new();
    for token in pgn.split_whitespace() {
        let san = token.rsplit('.').next().unwrap_or(token);
        if san.is_empty() {
            continue;
        }
        let mv = parse_san(&position, san)?;
        position.make_move(mv);
        moves.push(mv);
    }
    Ok(moves)
}

fn parse_san(position: &Position, san: &str) -> Result<Move, String> {
    let mut core = san;
    let mut suffix = "";
    while let Some(stripped) = core.strip_suffix(['!', '?']) {
        core = stripped;
    }
    if let Some(stripped) = core.strip_suffix(['+', '#']) {
        suffix = &core[core.len() - 1..];
        core = stripped;
    }
    if matches!(core, "O-O" | "0-0") {
        return castle(position, true, suffix);
    }
    if matches!(core, "O-O-O" | "0-0-0") {
        return castle(position, false, suffix);
    }
    let (piece, body) = match core.as_bytes().first() {
        Some(b'N') => (Piece::Knight, &core[1..]),
        Some(b'B') => (Piece::Bishop, &core[1..]),
        Some(b'R') => (Piece::Rook, &core[1..]),
        Some(b'Q') => (Piece::Queen, &core[1..]),
        Some(b'K') => (Piece::King, &core[1..]),
        Some(_) => (Piece::Pawn, core),
        None => return Err(format!("empty SAN: {san}")),
    };
    let (body, promotion) = match body.split_once('=') {
        Some((head, tail)) => {
            let piece = match tail.as_bytes().first() {
                Some(b'N') => Piece::Knight,
                Some(b'B') => Piece::Bishop,
                Some(b'R') => Piece::Rook,
                Some(b'Q') => Piece::Queen,
                _ => return Err(format!("bad promotion in {san}")),
            };
            (head, Some(piece))
        }
        None => (body, None),
    };
    let body = body.replace('x', "");
    if body.len() < 2 || !body.is_ascii() {
        return Err(format!("malformed SAN: {san}"));
    }
    let destination = parse_square(&body[body.len() - 2..])?;
    let hint = &body[..body.len() - 2];
    let mut candidates = position.legal_moves().into_iter().filter(|mv| {
        position
            .piece_at(mv.from)
            .is_some_and(|(_, moved)| moved == piece)
            && mv.to == destination
            && mv.promotion == promotion
            && hint_matches(hint, mv.from)
    });
    let mv = candidates
        .next()
        .ok_or_else(|| format!("no legal move for {san} in {}", position.to_fen()))?;
    if candidates.next().is_some() {
        return Err(format!("ambiguous SAN: {san}"));
    }
    let mut after = *position;
    after.make_move(mv);
    let check = after.is_in_check(after.side_to_move());
    match suffix {
        "#" if !(check && !after.has_legal_move()) => return Err(format!("bad mate in {san}")),
        "+" if !check => return Err(format!("bad check in {san}")),
        "" if check => return Err(format!("missing check marker in {san}")),
        _ => {}
    }
    Ok(mv)
}

fn castle(position: &Position, kingside: bool, suffix: &str) -> Result<Move, String> {
    let from = position.king_square(position.side_to_move());
    let to = if kingside { from + 2 } else { from - 2 };
    let mv = position
        .legal_moves()
        .into_iter()
        .find(|mv| mv.from == from && mv.to == to)
        .ok_or_else(|| format!("illegal castle in {}", position.to_fen()))?;
    let mut after = *position;
    after.make_move(mv);
    let check = after.is_in_check(after.side_to_move());
    if suffix == "+" && !check {
        return Err("bad check in castling".into());
    }
    Ok(mv)
}

fn hint_matches(hint: &str, from: u8) -> bool {
    match hint.as_bytes() {
        [] => true,
        [file] if file.is_ascii_alphabetic() => from % 8 == file - b'a',
        [rank] if rank.is_ascii_digit() => from / 8 == rank - b'1',
        _ => parse_square(hint) == Ok(from),
    }
}

/// Resolve a book move into the search result shape used by every backend.
/// The score is neutral: the book asserts a move, not an evaluation.
pub(crate) fn book_result(mv: BookMove) -> crate::SearchResult {
    crate::SearchResult {
        best_move: Some(mv.mv),
        score: 0,
        depth: 0,
        nodes: 0,
        qnodes: 0,
        elapsed: std::time::Duration::ZERO,
        pv: vec![mv.mv],
        stopped: false,
        hashfull: 0,
        book_move: Some(mv),
        selective: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(pgn: &str) -> OpeningBook {
        OpeningBook::from_pgn_tsv(&format!("eco\tname\tpgn\nX\tTest\t{pgn}\n")).unwrap()
    }

    #[test]
    fn replays_every_san_form_the_dataset_uses() {
        for (pgn, expected) in [
            ("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", "a7a6"),
            ("1. e4 c5 2. Nf3 d6 3. d4 cxd4", "c5d4"),
            ("1. d4 Nf6 2. c4 e6 3. Nc3 Bb4", "f8b4"),
            ("1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. Bb5 Bb4", "f8b4"),
            ("1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. e5 Nfd7", "f6d7"),
            ("1. e4 e5 2. Nf3 Nf6 3. Nxe5 d6", "d7d6"),
            ("1. e4 e5 2. Bc4 Nf6 3. Qh5 Nxe4", "f6e4"),
            ("1. Nf3 d5 2. e3 Nf6", "g8f6"),
            ("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6", "g8f6"),
            ("1. d4 d5 2. Nf3 Nf6 3. c4 e6 4. Nc3 Be7", "f8e7"),
            ("1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O", "e8g8"),
            ("1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6", "g8f6"),
        ] {
            let book = book(pgn);
            let moves = replay(pgn).unwrap();
            assert_eq!(moves.last().unwrap().to_string(), expected, "{pgn}");
            assert_eq!(book.lines(), 1, "{pgn}");
        }
    }

    #[test]
    fn rejects_illegal_ambiguous_and_wrong_markers() {
        for pgn in [
            "1. e4 e5 2. Kd3",            // illegal king move
            "1. e4 e5 2. Nf3+",           // phantom check marker
            "1. e4 e5 2. Ne2 Ne7 3. Nc3", // two knights reach c3
        ] {
            assert!(
                OpeningBook::from_pgn_tsv(&format!("e\tn\t{pgn}\n")).is_err(),
                "{pgn} must be rejected"
            );
        }
        assert!(replay("1. e4 exd5").is_err());
    }

    #[test]
    fn graph_shares_transpositions_and_names_the_deepest_line() {
        let book = OpeningBook::from_pgn_tsv(
            "A\tKing's Pawn\t1. e4\nB\tOpen Game\t1. e4 e5\nC\tOther\t1. d4 e5 2. e4\n",
        )
        .unwrap();
        let start = Position::startpos();
        let root = book.nodes.get(&start.hash()).unwrap();
        assert_eq!(root.total, 3);
        assert_eq!(root.moves[0].1, 2); // e4 is authored twice
        assert_eq!(root.moves[1].1, 1); // d4 once
        // The node reached after 1. e4 e5 is named by the line that ends there.
        let mut position = Position::startpos();
        for text in ["e2e4", "e7e5"] {
            position.make_move(position.parse_uci_move(text).unwrap());
        }
        assert_eq!(book.nodes.get(&position.hash()).unwrap().name, "Open Game");

        // Two move orders reaching one position share a single node.
        let transposed = OpeningBook::from_pgn_tsv(
            "A\tX\t1. e4 e5 2. Nf3 Nc6 3. d4\nB\tY\t1. e4 e5 2. d4 Nc6 3. Nf3\n",
        )
        .unwrap();
        let mut first = Position::startpos();
        for text in ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"] {
            first.make_move(first.parse_uci_move(text).unwrap());
        }
        let mut second = Position::startpos();
        for text in ["e2e4", "e7e5", "d2d4", "b8c6", "g1f3"] {
            second.make_move(second.parse_uci_move(text).unwrap());
        }
        assert_eq!(first.hash(), second.hash());
        assert!(transposed.nodes.contains_key(&first.hash()));
    }

    #[test]
    fn probe_is_deterministic_and_seeded() {
        let book = book("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6");
        let start = Position::startpos();
        // A lone authored move is always chosen, whatever the seed.
        assert_eq!(book.probe(&start, 0, 0).unwrap().uci(), "e2e4");
        assert_eq!(book.probe(&start, 0, 7).unwrap().uci(), "e2e4");
        // With several moves the same seed reproduces the same choice.
        let varied =
            OpeningBook::from_pgn_tsv("e\tn\t1. e4 e5\ne\tn\t1. d4 d5\ne\tn\t1. c4 e5\n").unwrap();
        assert_eq!(
            varied.probe(&start, 0, 42).unwrap().uci(),
            varied.probe(&start, 0, 42).unwrap().uci()
        );
    }

    #[test]
    fn weighted_choice_favors_more_authored_lines() {
        let book =
            OpeningBook::from_pgn_tsv("e\tn\t1. e4 e5\ne\tn\t1. e4 e5 2. Nf3\ne\tn\t1. d4 d5\n")
                .unwrap();
        let start = Position::startpos();
        let mut e4 = 0;
        let mut d4 = 0;
        for seed in 0..256 {
            match book.probe(&start, 0, seed).unwrap().uci().as_str() {
                "e2e4" => e4 += 1,
                "d2d4" => d4 += 1,
                other => panic!("unexpected {other}"),
            }
        }
        assert_eq!(e4 + d4, 256);
        assert!(e4 > d4 && d4 > 0, "e4={e4} d4={d4}");
    }

    #[test]
    fn exhausted_or_off_book_positions_return_none() {
        let book = book("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6");
        let mut position = Position::startpos();
        for text in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"] {
            position.make_move(position.parse_uci_move(text).unwrap());
        }
        assert!(book.probe(&position, 6, 0).is_none());
        assert!(book.probe(&Position::startpos(), 99, 0).is_none());
        assert!(
            book.probe(
                &Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap(),
                0,
                0
            )
            .is_none()
        );
    }
}
