//! Scheduling and batch evaluation contracts shared by CPU and GPU experiments.
use crate::{Position, SearchLimits, SearchResult, Searcher, eval::Evaluator};
use std::sync::{Arc, atomic::AtomicBool};

#[derive(Clone, Copy, Debug, Default)]
pub struct BatchTelemetry {
    pub positions: u64,
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
}
/// Optional execution-policy telemetry, separate from the model's identity.
#[derive(Clone, Copy, Debug)]
pub struct NeuralExecutionStats {
    pub mode: &'static str,
    pub max_cached_shapes: u32,
    pub first_used: bool,
    pub cached_shapes: u64,
    pub hits: u64,
    pub misses: u64,
    pub captures: u64,
    pub evictions: u64,
    pub retained_pool_bytes: u64,
    pub staging_logical_bytes: u64,
    pub peak_observed_pool_bytes: u64,
    pub capture_wall_ms: f64,
}
impl BatchTelemetry {
    pub fn accumulate(&mut self, other: Self) {
        self.positions += other.positions;
        self.h2d_ms += other.h2d_ms;
        self.kernel_ms += other.kernel_ms;
        self.d2h_ms += other.d2h_ms;
    }
}
/// Scores retain input order, use side-to-move centipawns, and exclude adjudication.
/// On failure outputs must be discarded. Device backends report failures explicitly.
pub trait BatchEvaluator: Send {
    fn name(&self) -> &str;
    fn evaluate_batch(
        &mut self,
        positions: &[Position],
        scores: &mut [i32],
    ) -> Result<BatchTelemetry, String>;
}
pub struct CpuBatchEvaluator {
    evaluator: Box<dyn Evaluator>,
}
impl CpuBatchEvaluator {
    pub fn new(evaluator: Box<dyn Evaluator>) -> Self {
        Self { evaluator }
    }
}
impl BatchEvaluator for CpuBatchEvaluator {
    fn name(&self) -> &str {
        self.evaluator.name()
    }
    fn evaluate_batch(
        &mut self,
        positions: &[Position],
        scores: &mut [i32],
    ) -> Result<BatchTelemetry, String> {
        if positions.len() != scores.len() {
            return Err("batch input/output length mismatch".into());
        }
        for (position, score) in positions.iter().zip(scores.iter_mut()) {
            self.evaluator.reset(position);
            *score = self.evaluator.evaluate(position);
        }
        Ok(BatchTelemetry {
            positions: positions.len() as u64,
            ..BatchTelemetry::default()
        })
    }
}
/// Search scheduling is independent from the evaluation backend and UCI transport.
pub trait SearchBackend: Send {
    fn name(&self) -> &str;
    fn evaluator_artifact_sha256(&self) -> Option<&str> {
        None
    }
    fn neural_execution_stats(&self) -> Result<Option<NeuralExecutionStats>, String> {
        Ok(None)
    }
    fn stop_handle(&self) -> Arc<AtomicBool>;
    fn clear(&mut self);
    fn search(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String>;
    fn batch_telemetry(&self) -> BatchTelemetry {
        BatchTelemetry::default()
    }
}
impl SearchBackend for Searcher {
    fn name(&self) -> &str {
        "alphabeta"
    }
    fn stop_handle(&self) -> Arc<AtomicBool> {
        Searcher::stop_handle(self)
    }
    fn clear(&mut self) {
        Searcher::clear(self);
    }
    fn search(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        Ok(Searcher::search(self, position, limits, history, report))
    }
}
