//! Reproducible CPU comparison of complete counterfactual attack relations.
use chen::codec::CodecPosition;
use chen::{
    Color, Piece, Position,
    core::{bishop_attacks, rook_attacks},
};
use std::{hint::black_box, time::Instant};

const KNIGHT: [u64; 64] = leapers(&[
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]);
const KING: [u64; 64] = leapers(&[
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]);

const fn leapers(offsets: &[(i8, i8)]) -> [u64; 64] {
    let mut table = [0; 64];
    let mut square = 0;
    while square < 64 {
        let mut index = 0;
        while index < offsets.len() {
            let file = (square % 8) as i8 + offsets[index].0;
            let rank = (square / 8) as i8 + offsets[index].1;
            if file >= 0 && file < 8 && rank >= 0 && rank < 8 {
                table[square] |= 1u64 << (rank * 8 + file);
            }
            index += 1;
        }
        square += 1;
    }
    table
}

// Produce the requested Boolean answer for every target by visiting sources
// once. This is a stronger orthodox control than 128 attackers_to() queries.
fn piece_union(position: &Position) -> [u64; 2] {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    const FILE_H: u64 = FILE_A << 7;
    let occupied = position.occupied();
    let mut attacks = [0; 2];
    for color in [Color::White, Color::Black] {
        let pawns = position.pieces(color, Piece::Pawn);
        let mut attack = if color == Color::White {
            ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9)
        } else {
            ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7)
        };
        for (piece, table) in [(Piece::Knight, &KNIGHT), (Piece::King, &KING)] {
            let mut sources = position.pieces(color, piece);
            while sources != 0 {
                let square = sources.trailing_zeros() as usize;
                sources &= sources - 1;
                attack |= table[square];
            }
        }
        let queens = position.pieces(color, Piece::Queen);
        let mut bishops = position.pieces(color, Piece::Bishop) | queens;
        while bishops != 0 {
            let square = bishops.trailing_zeros() as u8;
            bishops &= bishops - 1;
            attack |= bishop_attacks(square, occupied);
        }
        let mut rooks = position.pieces(color, Piece::Rook) | queens;
        while rooks != 0 {
            let square = rooks.trailing_zeros() as u8;
            rooks &= rooks - 1;
            attack |= rook_attacks(square, occupied);
        }
        attacks[color.index()] = attack;
    }
    attacks
}

fn verify(position: &Position, relation: &chen::codec::MoveRelation) {
    for (index, decision) in relation.decisions().iter().enumerate() {
        let mut child = *position;
        child.make_move(decision.mv);
        let union = piece_union(&child);
        for color in [Color::White, Color::Black] {
            let mut codec_mask = 0;
            let mut query_mask = 0;
            for square in 0..64 {
                if relation.attacked_after(index, color, square) {
                    codec_mask |= 1 << square;
                }
                if child.attackers_to(square, color) != 0 {
                    query_mask |= 1 << square;
                }
            }
            assert_eq!(
                union[color.index()],
                query_mask,
                "piece-union mask mismatch"
            );
            assert_eq!(codec_mask, query_mask, "codec mask mismatch");
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let count = args
        .get(1)
        .map_or(256, |s| s.parse::<usize>().expect("positions"));
    let repeats = args
        .get(2)
        .map_or(5, |s| s.parse::<usize>().expect("repeats"));
    assert!(count > 0 && repeats > 0);
    let mut corpus = Vec::with_capacity(count);
    let mut position = Position::startpos();
    let mut random = 0x8ac6_c0de_2026_1015u64;
    while corpus.len() < count {
        corpus.push(position);
        let moves = position.legal_moves();
        if moves.is_empty() || corpus.len() % 96 == 0 {
            position = Position::startpos();
        } else {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            position.make_move(moves[random as usize % moves.len()]);
        }
    }
    // Fixed candidate packets are shared by all timing paths. Scalar controls
    // receive these free; codec timing includes encoding and proposals.
    let candidates: Vec<_> = corpus
        .iter()
        .map(|p| CodecPosition::from_position(p).relation())
        .collect();
    for (position, relation) in corpus.iter().zip(&candidates) {
        verify(position, relation);
    }
    let attacks = |p: &Position| {
        let relation = CodecPosition::from_position(p).relation();
        let mut checksum = 0u64;
        for index in 0..relation.decisions().len() {
            for color in [Color::White, Color::Black] {
                for sq in 0..64 {
                    checksum =
                        checksum.wrapping_add(relation.attacked_after(index, color, sq) as u64);
                }
            }
        }
        black_box(checksum)
    };
    let scalar = |(p, relation): (&Position, &chen::codec::MoveRelation)| {
        let mut checksum = 0u64;
        for decision in relation.decisions() {
            let mut child = *p;
            child.make_move(decision.mv);
            for color in [Color::White, Color::Black] {
                for sq in 0..64 {
                    checksum = checksum.wrapping_add((child.attackers_to(sq, color) != 0) as u64);
                }
            }
        }
        black_box(checksum)
    };
    let union = |(p, relation): (&Position, &chen::codec::MoveRelation)| {
        // Retain every candidate's answers before consumption, as the codec
        // does. This control stores two conventional attack bitboards per lane.
        let retained: Vec<_> = relation
            .decisions()
            .iter()
            .map(|decision| {
                let mut child = *p;
                child.make_move(decision.mv);
                piece_union(&child)
            })
            .collect();
        let mut checksum = 0u64;
        for masks in retained {
            for mask in masks {
                // Consume exactly the same 128 color/target Boolean answers.
                for square in 0..64 {
                    checksum = checksum.wrapping_add(u64::from(mask & (1 << square) != 0));
                }
            }
        }
        black_box(checksum)
    };
    let codec_once: u64 = corpus.iter().map(attacks).sum();
    let scalar_once: u64 = corpus.iter().zip(&candidates).map(scalar).sum();
    assert_eq!(
        codec_once, scalar_once,
        "full attack relation oracle mismatch"
    );
    let mut codec_checksum = 0;
    let started = Instant::now();
    for _ in 0..repeats {
        codec_checksum += corpus.iter().map(attacks).sum::<u64>();
    }
    let codec_time = started.elapsed();
    let mut scalar_checksum = 0;
    let started = Instant::now();
    for _ in 0..repeats {
        scalar_checksum += corpus.iter().zip(&candidates).map(scalar).sum::<u64>();
    }
    let scalar_time = started.elapsed();
    assert_eq!(codec_checksum, scalar_checksum);
    let mut union_checksum = 0;
    let started = Instant::now();
    for _ in 0..repeats {
        union_checksum += corpus.iter().zip(&candidates).map(union).sum::<u64>();
    }
    let union_time = started.elapsed();
    assert_eq!(codec_checksum, union_checksum);
    let mut legal_checksum = 0;
    let started = Instant::now();
    for _ in 0..repeats {
        for p in &corpus {
            legal_checksum += black_box(p).legal_moves().len();
        }
    }
    let legal_time = started.elapsed();
    let decisions: usize = candidates.iter().map(|r| r.decisions().len()).sum();
    let words: usize = candidates.iter().map(|r| r.stats().lane_words).sum();
    let edits: usize = candidates.iter().map(|r| r.stats().patch_edits).sum();
    let bytes: usize = candidates.iter().map(|r| r.stats().attack_bytes).sum();
    let denominator = (count * repeats) as f64;
    println!("positions={count} repeats={repeats} seed=8ac6c0de20261015");
    println!(
        "candidates={decisions} lane_words={words} sparse_edits={edits} retained_attack_bytes={bytes}"
    );
    println!(
        "codec_full_relation_ns_per_position={:.1}",
        codec_time.as_nanos() as f64 / denominator
    );
    println!(
        "scalar_queries_full_relation_ns_per_position={:.1}",
        scalar_time.as_nanos() as f64 / denominator
    );
    println!(
        "scalar_queries_over_codec={:.3}",
        scalar_time.as_secs_f64() / codec_time.as_secs_f64()
    );
    println!(
        "scalar_piece_union_full_relation_ns_per_position={:.1}",
        union_time.as_nanos() as f64 / denominator
    );
    println!(
        "codec_over_scalar_piece_union={:.3}",
        codec_time.as_secs_f64() / union_time.as_secs_f64()
    );
    println!(
        "orthodox_legal_only_ns_per_position={:.1} (smaller workload)",
        legal_time.as_nanos() as f64 / denominator
    );
    println!("attack_checksum={codec_checksum} legal_checksum={legal_checksum}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_union_masks_match_all_queries_on_special_move_children() {
        for fen in [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
            "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
            "7k/8/8/Q1Q1Q1Q1/8/Q1Q1Q1Q1/8/K7 w - - 0 1",
        ] {
            let position = Position::from_fen(fen).unwrap();
            let relation = CodecPosition::from_position(&position).relation();
            verify(&position, &relation);
        }
    }
}
