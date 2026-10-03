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
    pub retained_budget_bytes: u64,
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
/// Per-iteration bounded GPU work, separate from search-value publication.
#[derive(Clone, Debug, Default)]
pub struct BoundedIterationStats {
    pub requested_depth: u32,
    pub kind: u32,
    pub abort_reason: u32,
    pub nodes: u64,
    pub qnodes: u64,
    pub elapsed_ticks: u64,
    pub wall_ms: f64,
}
#[derive(Clone, Debug, Default)]
pub struct BoundedSearchStats {
    pub bootstrap_nodes: u64,
    pub history_upload_ms: f64,
    pub wall_clock_khz: u64,
    pub scratch_bytes: u64,
    pub node_limit: u64,
    pub quiescence: bool,
    pub completed_depth: u8,
    pub stop_reason: &'static str,
    pub iterations: Vec<BoundedIterationStats>,
}
impl BoundedSearchStats {
    #[cfg(feature = "hip")]
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "bootstrap_nodes": self.bootstrap_nodes,
            "history_upload_ms": self.history_upload_ms,
            "wall_clock_khz": self.wall_clock_khz,
            "scratch_bytes": self.scratch_bytes,
            "node_limit": self.node_limit,
            "quiescence": self.quiescence,
            "completed_depth": self.completed_depth,
            "stop_reason": self.stop_reason,
            "ordering": "canonical", "tt": false,
            "iterations": self.iterations.iter().map(|i| serde_json::json!({
                "requested_depth":i.requested_depth,"kind":i.kind,"abort_reason":i.abort_reason,
                "nodes":i.nodes,"qnodes":i.qnodes,"elapsed_ticks":i.elapsed_ticks,"wall_ms":i.wall_ms
            })).collect::<Vec<_>>()
        }).to_string()
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
    fn drowsy_stats(&self) -> Option<crate::drowsy::DrowsyStats> {
        None
    }
    fn root_pool_stats(&self) -> Option<String> {
        None
    }
    fn bounded_stats(&self) -> Option<BoundedSearchStats> {
        None
    }
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
