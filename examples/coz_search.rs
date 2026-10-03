//! Stationary fixed-node CPU workload for Coz; no move-clock or GPU inference.
use chen::{Position, SearchConfig, SearchLimits, Searcher, Tapered};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let iterations = args
        .first()
        .map_or(Ok(80), |v| v.parse::<u32>())
        .map_err(|_| "iterations must be an integer")?;
    let nodes = args
        .get(1)
        .map_or(Ok(200_000), |v| v.parse::<u64>())
        .map_err(|_| "nodes must be an integer")?;
    if args.len() > 2 || !(1..=1000).contains(&iterations) || !(256..=500_000).contains(&nodes) {
        return Err("usage: coz_search [iterations1..1000] [nodes256..500000]".into());
    }
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let original = Position::from_fen(fen)?;
    let mut searcher = Searcher::new(Box::new(Tapered), SearchConfig::default());
    let mut reference = None;
    for _ in 0..iterations {
        searcher.clear();
        let mut position = original;
        coz::begin!("cpu_fixed_node_search");
        let result = searcher.search(
            &mut position,
            SearchLimits {
                depth: 64,
                nodes: Some(nodes),
                time: None,
            },
            &[],
            |_| {},
        );
        coz::end!("cpu_fixed_node_search");
        let exact = (
            result.depth,
            result.score,
            result.best_move,
            result.nodes,
            result.qnodes,
            result.pv,
        );
        if result.nodes != nodes
            || position.to_fen() != fen
            || reference.as_ref().is_some_and(|r| r != &exact)
        {
            return Err("stationary fixed-node trajectory changed".into());
        }
        reference = Some(exact);
        coz::progress!("cpu_completed_fixed_node_search");
    }
    println!("completed {iterations} identical searches, {nodes} visited nodes each");
    Ok(())
}
