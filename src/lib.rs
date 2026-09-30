//! Chen: a chess rules kernel, pluggable evaluation, and reproducible search.
pub mod backend;
pub mod core;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod device;
pub mod eval;
pub mod frontier;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod hip;
pub mod resident;
pub mod rules;
pub mod search;
pub mod strength;
pub mod tt;

pub use core::{Color, Move, Piece, Position, Undo, perft};
pub use eval::{Evaluator, Material, Tapered};
pub use search::{SearchConfig, SearchLimits, SearchResult, Searcher};
