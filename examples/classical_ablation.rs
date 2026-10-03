//! Bounded fresh-searcher controls; timings are fixed-depth process observations.
use chen::{Position, SearchConfig, SearchLimits, Searcher, Tapered};
use std::collections::HashMap;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let depth: u8 = args.get(1).expect("depth required").parse().unwrap();
    let rounds: usize = args.get(2).expect("rounds required").parse().unwrap();
    let suite = args.get(3).expect("suite required");
    assert!((1..=8).contains(&depth) && (1..=8).contains(&rounds));
    let text = std::fs::read_to_string(suite).unwrap();
    let positions: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (name, fen) = line.split_once('|').unwrap();
            (name.trim(), fen.trim())
        })
        .collect();
    let mut expected = HashMap::new();
    println!(
        "round,position,policy,requested_depth,depth,score,bestmove,nodes,qnodes,elapsed_ns,pvs_probes,pvs_researches,aspiration_retries"
    );
    for round in 0..rounds {
        for &(name, fen) in &positions {
            for offset in 0..4 {
                let variant = (round + offset) % 4;
                let (policy, pvs, aspiration_window) = match variant {
                    0 => ("baseline", false, 0),
                    1 => ("pvs", true, 0),
                    2 => ("aspiration25", false, 25),
                    _ => ("pvs_aspiration25", true, 25),
                };
                let mut p = Position::from_fen(fen).unwrap();
                let before = p;
                let mut searcher = Searcher::new(
                    Box::new(Tapered),
                    SearchConfig {
                        pvs,
                        aspiration_window,
                        ..SearchConfig::default()
                    },
                );
                let r = searcher.search(
                    &mut p,
                    SearchLimits {
                        depth,
                        nodes: Some(2_000_000),
                        time: None,
                    },
                    &[],
                    |_| {},
                );
                assert!(
                    !r.stopped,
                    "fixed-depth node safety bound reached: {name}/{policy}"
                );
                assert_eq!(p, before);
                if variant == 0 {
                    expected.insert(name, (r.depth, r.score));
                }
                assert_eq!(
                    expected.get(name),
                    Some(&(r.depth, r.score)),
                    "{name}/{policy}"
                );
                for &mv in &r.pv {
                    assert!(p.legal_moves().contains(&mv));
                    p.make_move(mv);
                }
                let s = searcher.stats();
                println!(
                    "{round},{name},{policy},{depth},{},{},{},{},{},{},{},{},{}",
                    r.depth,
                    r.score,
                    r.best_move.map_or("0000".into(), |m| m.to_string()),
                    r.nodes,
                    r.qnodes,
                    r.elapsed.as_nanos(),
                    s.pvs_probes,
                    s.pvs_researches,
                    s.aspiration_retries
                );
            }
        }
    }
}
