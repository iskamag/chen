//! Isolated candidate verification and representative CPU search/perft workload.
use chen::{Position, SearchConfig, SearchLimits, Searcher, Tapered, perft};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
#[path = "../src/attacks.rs"]
mod candidate;

fn walk(square: u8, occupied: u64, diagonal: bool) -> u64 {
    let mut result = 0;
    for dx in -1i8..=1 {
        for dy in -1i8..=1 {
            if (dx == 0 && dy == 0) || (diagonal != (dx != 0 && dy != 0)) {
                continue;
            }
            let mut x = (square % 8) as i8 + dx;
            let mut y = (square / 8) as i8 + dy;
            while (0..8).contains(&x) && (0..8).contains(&y) {
                let bit = 1u64 << (y * 8 + x);
                result |= bit;
                if occupied & bit != 0 {
                    break;
                }
                x += dx;
                y += dy;
            }
        }
    }
    result
}

fn verify() {
    let mut checks = 0u64;
    for diagonal in [false, true] {
        for square in 0u8..64 {
            let relevant = walk(square, 0, diagonal);
            let bits: Vec<_> = (0..64).filter(|&s| relevant & (1u64 << s) != 0).collect();
            for subset in 0..(1u32 << bits.len()) {
                let mut occ = 0;
                for (i, &s) in bits.iter().enumerate() {
                    if subset & (1 << i) != 0 {
                        occ |= 1u64 << s;
                    }
                }
                for origin in [false, true] {
                    let occupancy = occ | if origin { 1u64 << square } else { 0 };
                    let wanted = walk(square, occupancy, diagonal);
                    let got = if diagonal {
                        candidate::bishop(square, occupancy)
                    } else {
                        candidate::rook(square, occupancy)
                    };
                    let core = if diagonal {
                        chen::core::bishop_attacks(square, occupancy)
                    } else {
                        chen::core::rook_attacks(square, occupancy)
                    };
                    assert_eq!(
                        got, wanted,
                        "square={square} occ={occupancy:x} diagonal={diagonal}"
                    );
                    assert_eq!(core, wanted);
                    checks += 2;
                }
            }
        }
    }
    let mut random = 0x0c4e_1030_2026_1002u64;
    for i in 0..100_000 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let square = (i % 64) as u8;
        for diagonal in [false, true] {
            let wanted = walk(square, random, diagonal);
            let got = if diagonal {
                candidate::bishop(square, random)
            } else {
                candidate::rook(square, random)
            };
            let core = if diagonal {
                chen::core::bishop_attacks(square, random)
            } else {
                chen::core::rook_attacks(square, random)
            };
            assert_eq!(got, wanted);
            assert_eq!(core, wanted);
            checks += 2;
        }
    }
    println!("{{\"phase\":\"exhaustive\",\"checks\":{checks},\"status\":\"passed\"}}");
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).expect("verify|perft|nodes|timed");
    if mode == "verify" {
        verify();
        return;
    }
    let suite = std::fs::read_to_string(args.get(2).expect("suite path")).unwrap();
    for line in suite
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
    {
        let (name, fen) = line.split_once('|').expect("name|FEN");
        let mut position = Position::from_fen(fen.trim()).unwrap();
        if mode == "perft" {
            let original = position;
            let started = Instant::now();
            let depth = if name.trim() == "start" { 5 } else { 4 };
            let nodes = black_box(perft(&mut position, depth));
            assert_eq!(position, original);
            println!(
                "{{\"phase\":\"perft\",\"position\":\"{}\",\"depth\":{depth},\"nodes\":{nodes},\"elapsed_ns\":{}}}",
                name.trim(),
                started.elapsed().as_nanos()
            );
        } else {
            assert!(mode == "nodes" || mode == "timed");
            let mut searcher = Searcher::new(Box::new(Tapered), SearchConfig::default());
            let result = searcher.search(
                &mut position,
                SearchLimits {
                    depth: 64,
                    nodes: if mode == "nodes" { Some(200_000) } else { None },
                    time: if mode == "timed" {
                        Some(Duration::from_secs(1))
                    } else {
                        None
                    },
                },
                &[],
                |_| {},
            );
            let best = result
                .best_move
                .map_or_else(|| "0000".into(), |m| m.to_string());
            let pv = result
                .pv
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "{{\"phase\":\"{mode}\",\"position\":\"{}\",\"depth\":{},\"nodes\":{},\"qnodes\":{},\"score_cp\":{},\"bestmove\":\"{best}\",\"pv\":\"{pv}\",\"stopped\":{},\"elapsed_ns\":{}}}",
                name.trim(),
                result.depth,
                result.nodes,
                result.qnodes,
                result.score,
                result.stopped,
                result.elapsed.as_nanos()
            );
        }
    }
}
