//! CPU-only paired profile: frozen c4c9096 arithmetic versus current tapered evaluator.
//! cargo run --release --example tapered_profile -- micro [passes]
//! cargo run --release --example tapered_profile -- search [depth] [rounds]
use chen::{Evaluator, Position, SearchConfig, SearchLimits, SearchResult, Searcher, Tapered};
use std::{hint::black_box, time::Instant};
#[path = "../benches/support/tapered_reference.rs"]
mod reference;
use reference::ArithmeticTapered;

fn suite() -> Vec<(&'static str, Position)> {
    include_str!("../benches/positions.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.split_once('|'))
        .map(|(name, fen)| (name.trim(), Position::from_fen(fen.trim()).unwrap()))
        .collect()
}

fn evaluator(reference: bool) -> Box<dyn Evaluator> {
    if reference {
        Box::new(ArithmeticTapered)
    } else {
        Box::new(Tapered)
    }
}

fn label(reference: bool) -> &'static str {
    if reference { "arithmetic" } else { "lookup" }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("micro") {
        "micro" => {
            let passes: usize = args.get(2).map_or(5000, |s| s.parse().unwrap());
            let mut positions = Vec::new();
            let mut random = 0xc4e0_2026_1015_u64;
            for (_, mut p) in suite() {
                for _ in 0..256 {
                    positions.push(p);
                    let moves = p.legal_moves();
                    if moves.is_empty() {
                        break;
                    }
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    p.make_move(moves[random as usize % moves.len()]);
                }
            }
            for p in &positions {
                assert_eq!(Tapered.evaluate(p), ArithmeticTapered.evaluate(p));
            }
            for round in 0..4 {
                let mut checksums = [0; 2];
                for reference in if round % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                } {
                    let mut eval = evaluator(reference);
                    let started = Instant::now();
                    let mut sum = 0i64;
                    for _ in 0..passes {
                        for p in &positions {
                            sum += black_box(eval.evaluate(black_box(p))) as i64;
                        }
                    }
                    checksums[reference as usize] = sum;
                    println!(
                        "micro round={round} evaluator={} positions={} evaluations={} elapsed_ns={} checksum={sum}",
                        label(reference),
                        positions.len(),
                        positions.len() * passes,
                        started.elapsed().as_nanos()
                    );
                }
                assert_eq!(checksums[0], checksums[1]);
            }
        }
        "search" => {
            let depth: u8 = args.get(2).map_or(5, |s| s.parse().unwrap());
            let rounds: usize = args.get(3).map_or(3, |s| s.parse().unwrap());
            for round in 0..rounds {
                for (name, position) in suite() {
                    let mut results: [Option<SearchResult>; 2] = [None, None];
                    for reference in if round % 2 == 0 {
                        [true, false]
                    } else {
                        [false, true]
                    } {
                        let mut searcher =
                            Searcher::new(evaluator(reference), SearchConfig::default());
                        let mut p = position;
                        let result = searcher.search(
                            &mut p,
                            SearchLimits {
                                depth,
                                nodes: None,
                                time: None,
                            },
                            &[],
                            |_| {},
                        );
                        assert_eq!(p, position);
                        println!(
                            "search round={round} position={name} evaluator={} depth={} nodes={} qnodes={} score={} bestmove={} elapsed_ns={}",
                            label(reference),
                            result.depth,
                            result.nodes,
                            result.qnodes,
                            result.score,
                            result
                                .best_move
                                .map_or_else(|| "0000".into(), |m| m.to_string()),
                            result.elapsed.as_nanos()
                        );
                        results[reference as usize] = Some(result);
                    }
                    let a = results[0].as_ref().unwrap();
                    let b = results[1].as_ref().unwrap();
                    assert_eq!(
                        (a.best_move, a.score, a.depth, a.nodes, a.qnodes, &a.pv),
                        (b.best_move, b.score, b.depth, b.nodes, b.qnodes, &b.pv)
                    );
                }
            }
        }
        mode => panic!("unknown mode {mode}: use micro or search"),
    }
}
