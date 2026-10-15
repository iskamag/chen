//! A searchless consumer of the codec's exact counterfactual relation.
//!
//! Legality and attack answers are exact. Move preferences are a deliberately
//! simple, untrained static heuristic, not minimax values or tactical proofs.
use crate::{
    Color, Move, OpeningBook, Piece, Position, SearchLimits, SearchResult,
    backend::SearchBackend,
    codec::{CodecPosition, MoveRelation, Patch},
    core::PIECES,
    eval::{VALUES, bishop_pair_terms, piece_terms, taper},
    opening::book_result,
    search::{MATE, MAX_EVAL},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct CodecPolicyStats {
    pub score_kind: &'static str,
    pub alternatives: usize,
    pub legal_alternatives: usize,
    pub scored_alternatives: usize,
    pub lane_words: usize,
    pub boundary_compositions: usize,
    pub patch_edits: usize,
    pub attack_bytes: usize,
}

/// Inspectable preference for one legal alternative, in the root mover's frame.
#[derive(Clone, Copy, Debug)]
pub struct Preference {
    pub mv: Move,
    pub static_cp: i32,
    pub own_exposure: i32,
    pub enemy_exposure: i32,
    pub gives_check: bool,
    pub score: i32,
}

/// Aggregate sufficient to update the existing Tapered evaluator by sparse edits.
/// Phase stays uncapped until the final reduction (promotions can exceed 24).
struct Terms {
    mg: i32,
    eg: i32,
    phase: i32,
    bishops: [i32; 2],
}
impl Terms {
    fn from_position(position: &Position) -> Self {
        let mut terms = Self {
            mg: 0,
            eg: 0,
            phase: 0,
            bishops: [0; 2],
        };
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                let mut squares = position.pieces(color, piece);
                while squares != 0 {
                    let square = squares.trailing_zeros() as u8;
                    squares &= squares - 1;
                    terms.add(color, piece, square, 1);
                }
            }
        }
        terms
    }
    fn add(&mut self, color: Color, piece: Piece, square: u8, sign: i32) {
        let (mg, eg, phase) = piece_terms(color, piece, square);
        self.mg += sign * mg;
        self.eg += sign * eg;
        self.phase += sign * phase;
        if piece == Piece::Bishop {
            self.bishops[color.index()] += sign;
        }
    }
    fn after(&self, board: &[Option<(Color, Piece)>; 64], patch: &Patch) -> i32 {
        let mut after = Self {
            mg: self.mg,
            eg: self.eg,
            phase: self.phase,
            bishops: self.bishops,
        };
        for edit in patch.edits() {
            if let Some((color, piece)) = board[edit.square as usize] {
                after.add(color, piece, edit.square, -1);
            }
            if let Some((color, piece)) = edit.piece {
                after.add(color, piece, edit.square, 1);
            }
        }
        for color in [Color::White, Color::Black] {
            let (mg, eg) = bishop_pair_terms(color, after.bishops[color.index()] as u32);
            after.mg += mg;
            after.eg += eg;
        }
        taper(after.mg, after.eg, after.phase)
    }
}

/// Scores alternatives by patch deltas and queries retained attack channels.
/// It never makes a child Position and never requests the conventional move generator.
pub struct Preferences<'a> {
    position: &'a Position,
    relation: &'a MoveRelation,
    board: [Option<(Color, Piece)>; 64],
    terms: Terms,
}
impl<'a> Preferences<'a> {
    /// `relation` must have been composed from this same position.
    pub fn new(position: &'a Position, relation: &'a MoveRelation) -> Self {
        Self {
            position,
            relation,
            board: std::array::from_fn(|square| position.piece_at(square as u8)),
            terms: Terms::from_position(position),
        }
    }
    pub fn score(&self, index: usize) -> Option<Preference> {
        let decision = &self.relation.decisions()[index];
        if !decision.legal {
            return None;
        }
        let mover = self.position.side_to_move();
        let white_cp = self.terms.after(&self.board, &decision.patch);
        let static_cp = if mover == Color::White {
            white_cp
        } else {
            -white_cp
        };
        let mut exposure = [0; 2];
        for square in 0..64u8 {
            let piece = decision
                .patch
                .edits()
                .iter()
                .find(|edit| edit.square == square)
                .map_or(self.board[square as usize], |edit| edit.piece);
            let Some((color, piece)) = piece else {
                continue;
            };
            if piece == Piece::King
                || !self
                    .relation
                    .attacked_after(index, color.opposite(), square)
            {
                continue;
            }
            // Geometric pressure, not legal exchange value: pinned attackers also
            // attack squares, and a defended piece can still lose an exchange.
            let value = VALUES[piece.index()];
            let cost = if self.relation.attacked_after(index, color, square) {
                value / 8
            } else {
                value
            };
            exposure[color.index()] = exposure[color.index()].max(cost);
        }
        let own_exposure = exposure[mover.index()];
        let enemy_exposure = exposure[mover.opposite().index()];
        // The opponent moves next: charge our worst exposure in full, credit
        // prospective pressure only partially. These constants are not tuned.
        let score = (static_cp - own_exposure
            + enemy_exposure / 4
            + if decision.gives_check { 12 } else { 0 })
        .clamp(-MAX_EVAL, MAX_EVAL);
        Some(Preference {
            mv: decision.mv,
            static_cp,
            own_exposure,
            enemy_exposure,
            gives_check: decision.gives_check,
            score,
        })
    }
}

#[derive(Default)]
pub struct CodecPolicy {
    stop: Arc<AtomicBool>,
    stats: CodecPolicyStats,
    book: Option<(Arc<OpeningBook>, u64, usize)>,
}
impl CodecPolicy {
    pub fn stats(&self) -> CodecPolicyStats {
        self.stats
    }
}
impl SearchBackend for CodecPolicy {
    fn name(&self) -> &str {
        "codec-policy"
    }
    fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    fn clear(&mut self) {
        self.stats = CodecPolicyStats::default();
    }
    fn codec_stats(&self) -> Option<CodecPolicyStats> {
        Some(self.stats)
    }
    fn set_book(&mut self, book: Option<Arc<OpeningBook>>, seed: u64, plies: usize) {
        self.book = book.map(|book| (book, seed, plies.clamp(1, 64)));
    }
    fn search(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        let started = Instant::now();
        self.clear();
        let relation = CodecPosition::from_position(position).relation();
        let stats = relation.stats();
        let legal_count = relation
            .decisions()
            .iter()
            .filter(|decision| decision.legal)
            .count();
        self.stats = CodecPolicyStats {
            score_kind: "fallback",
            alternatives: stats.candidates,
            legal_alternatives: legal_count,
            lane_words: stats.lane_words,
            boundary_compositions: stats.boundary_compositions,
            patch_edits: stats.patch_edits,
            attack_bytes: stats.attack_bytes,
            scored_alternatives: 0,
        };
        let fallback = relation
            .decisions()
            .iter()
            .find(|decision| decision.legal)
            .map(|decision| decision.mv);
        let mut result = SearchResult {
            best_move: fallback,
            score: 0,
            depth: 0,
            nodes: 0,
            qnodes: 0,
            elapsed: started.elapsed(),
            pv: fallback.into_iter().collect(),
            stopped: false,
            hashfull: 0,
            book_move: None,
            selective: None,
        };
        if fallback.is_none() {
            self.stats.score_kind = "root_terminal";
            result.score = if relation.attacked_before(
                position.side_to_move().opposite(),
                position.king_square(position.side_to_move()),
            ) {
                -MATE
            } else {
                0
            };
        } else if position.halfmove_clock() >= 100
            || position.insufficient_material()
            || history
                .iter()
                .rev()
                .take(position.halfmove_clock() as usize)
                .filter(|&&hash| hash == position.hash())
                .count()
                >= 2
        {
            // Root terminals precede preferences and opening-book selection.
            self.stats.score_kind = "root_terminal";
        } else if let Some((book, seed, plies)) = &self.book
            && history.len() < *plies
            && let Some(entry) = book.probe(position, history.len(), *seed)
            && relation
                .decisions()
                .iter()
                .any(|d| d.legal && d.mv == entry.mv)
        {
            self.stats.score_kind = "book";
            result = book_result(entry);
        } else {
            let preferences = Preferences::new(position, &relation);
            let mut best = i32::MIN;
            for index in 0..relation.decisions().len() {
                if self.stop.load(Ordering::Relaxed)
                    || limits.time.is_some_and(|limit| started.elapsed() >= limit)
                {
                    result.stopped = true;
                    break;
                }
                let Some(preference) = preferences.score(index) else {
                    continue;
                };
                self.stats.scored_alternatives += 1;
                self.stats.score_kind = "static_heuristic";
                if preference.score > best {
                    best = preference.score;
                    result.best_move = Some(preference.mv);
                    result.score = preference.score;
                    result.pv = vec![preference.mv];
                }
            }
        }
        result.elapsed = started.elapsed();
        report(&result);
        Ok(result)
    }
}
