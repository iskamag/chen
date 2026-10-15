//! Reproducible CPU comparison of complete counterfactual attack relations.
use chen::codec::CodecPosition;
use chen::{Color, Position};
use std::{hint::black_box, time::Instant};

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
    // Fixed candidate packets are shared by both timing paths. The scalar oracle
    // is given these free, making its full-attack comparison conservative.
    let candidates: Vec<_> = corpus
        .iter()
        .map(|p| CodecPosition::from_position(p).relation())
        .collect();
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
        "scalar_full_relation_ns_per_position={:.1}",
        scalar_time.as_nanos() as f64 / denominator
    );
    println!(
        "scalar_over_codec={:.3}",
        scalar_time.as_secs_f64() / codec_time.as_secs_f64()
    );
    println!(
        "orthodox_legal_only_ns_per_position={:.1} (smaller workload)",
        legal_time.as_nanos() as f64 / denominator
    );
    println!("attack_checksum={codec_checksum} legal_checksum={legal_checksum}");
}
