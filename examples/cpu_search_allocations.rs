//! Fixed-work CPU search control. Build unchanged against baseline and candidate.
use chen::{Position, SearchConfig, SearchLimits, Searcher, Tapered};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicU64, Ordering},
};

struct CountAlloc;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
// Counting is outside search policy; both binaries use this same allocator.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn main() -> Result<(), String> {
    let nodes = std::env::args()
        .nth(1)
        .map_or(Ok(200_000), |s| s.parse::<u64>())
        .map_err(|_| "node budget must be an integer")?;
    if nodes == 0 {
        return Err("node budget must be positive".into());
    }
    for (name, fen) in [
        (
            "start",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
        ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        (
            "middlegame",
            "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 2 8",
        ),
    ] {
        for pvs in [false, true] {
            let mut position = Position::from_fen(fen)?;
            let before = position;
            let mut searcher = Searcher::new(
                Box::new(Tapered),
                SearchConfig {
                    pvs,
                    ..SearchConfig::default()
                },
            );
            let allocations = ALLOCATIONS.load(Ordering::Relaxed);
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
            let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations;
            assert_eq!(position, before);
            println!(
                "{name} pvs={pvs} ns={} allocations={allocations} nodes={} qnodes={} depth={} score={} best={} pv={}",
                result.elapsed.as_nanos(),
                result.nodes,
                result.qnodes,
                result.depth,
                result.score,
                result
                    .best_move
                    .map_or_else(|| "-".into(), |m| m.to_string()),
                result
                    .pv
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }
    Ok(())
}
