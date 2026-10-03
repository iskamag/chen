//! Chen: a chess rules kernel, pluggable evaluation, and reproducible search.
pub mod attacks;
pub mod backend;
pub mod catnip;
pub mod core;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod device;
pub mod eval;
pub mod frontier;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod hip;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod hip_bounded;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod hip_root_pool;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod hip_root_retry;
#[cfg(all(feature = "hip", target_os = "linux"))]
pub mod neural;
pub mod resident;
pub mod rules;
pub mod search;
pub mod selective;
pub mod strength;
pub mod tt;
pub mod two_fidelity;

pub use core::{Color, Move, Piece, Position, Undo, perft};
pub use eval::{Evaluator, Material, Tapered};
pub use search::{SearchConfig, SearchLimits, SearchResult, Searcher};
