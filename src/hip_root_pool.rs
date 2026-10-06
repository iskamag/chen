//! Experimental concurrent root children with resumable device DFS epochs.
use crate::{
    Move, Position, SearchLimits, SearchResult,
    backend::{BatchTelemetry, SearchBackend},
    hip::{Library, buffer_string},
    rules::{RuleMove, RuleState},
};
use std::{
    ffi::{c_char, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Job {
    root: RuleState,
    node_budget: u64,
    history_offset: u64,
    history_count: u64,
    abi_version: u32,
    struct_size: u32,
    depth: u32,
    max_qply: u32,
    evaluator: u32,
    quiescence: u32,
    root_ply: u32,
    reserved: u32,
    alpha: i32,
    beta: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Packet {
    abi_version: u32,
    struct_size: u32,
    kind: u32,
    abort_reason: u32,
    score: i32,
    completed_depth: u32,
    pv_length: u32,
    reserved: u32,
    nodes: u64,
    qnodes: u64,
    pv: [RuleMove; 64],
}
impl Default for Packet {
    fn default() -> Self {
        Self {
            abi_version: 0,
            struct_size: 0,
            kind: 0,
            abort_reason: 0,
            score: 0,
            completed_depth: 0,
            pv_length: 0,
            reserved: 0,
            nodes: 0,
            qnodes: 0,
            pv: [RuleMove::default(); 64],
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Snapshot {
    published: Packet,
    root_hash: u64,
    root_epoch: u64,
    nodes: u64,
    qnodes: u64,
    iteration_nodes: u64,
    elapsed_ticks: u64,
    abi_version: u32,
    struct_size: u32,
    requested_depth: u32,
    root_count: u32,
    completed_children: u32,
    exact_children: u32,
    upper_children: u32,
    paused_children: u32,
    abort_reason: u32,
    iteration_complete: u32,
    terminal: u32,
    reserved: u32,
    fallback: RuleMove,
    padding: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
    wall_clock_khz: u64,
    scratch_bytes: u64,
    history_capacity: u64,
}
type Create = unsafe extern "C" fn(i32, *mut c_char, u64) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Cancel = unsafe extern "C" fn(*mut c_void);
type Reset = unsafe extern "C" fn(
    *mut c_void,
    *const Job,
    *const u64,
    u64,
    u64,
    *mut Snapshot,
    *mut c_char,
    u64,
) -> i32;
type Begin = unsafe extern "C" fn(*mut c_void, u32, *mut Snapshot, *mut c_char, u64) -> i32;
type Advance = unsafe extern "C" fn(
    *mut c_void,
    u64,
    *mut Snapshot,
    *mut c_void,
    u32,
    *mut Stats,
    *mut c_char,
    u64,
) -> i32;
struct Native {
    library: Library,
    context: NonNull<c_void>,
    destroy: Destroy,
    cancel: Cancel,
    reset: Reset,
    begin: Begin,
    advance: Advance,
}
// SAFETY: unique serialized context; only mapped atomic cancellation callback
// may overlap advance, and its scoped monitor joins before context/library drop.
unsafe impl Send for Native {}
impl Native {
    fn new(path: &Path) -> Result<Self, String> {
        let library = Library::open(path)?;
        macro_rules! symbol {
            ($n:literal,$t:ty) => {{ unsafe { std::mem::transmute::<*mut c_void, $t>(library.symbol($n)?) } }};
        }
        let version = symbol!(c"chen_root_pool_abi_version", unsafe extern "C" fn() -> u32);
        if unsafe { version() } != 3 {
            return Err("expected root pool ABI3".into());
        }
        let create = symbol!(c"chen_root_pool_create", Create);
        let destroy = symbol!(c"chen_root_pool_destroy", Destroy);
        let cancel = symbol!(c"chen_root_pool_cancel", Cancel);
        let reset = symbol!(c"chen_root_pool_reset", Reset);
        let begin = symbol!(c"chen_root_pool_begin", Begin);
        let advance = symbol!(c"chen_root_pool_advance", Advance);
        let mut error = [0 as c_char; 1024];
        let context = NonNull::new(unsafe { create(0, error.as_mut_ptr(), 1024) })
            .ok_or_else(|| buffer_string(&error))?;
        Ok(Self {
            library,
            context,
            destroy,
            cancel,
            reset,
            begin,
            advance,
        })
    }
    fn reset(&mut self, job: Job, history: &[u64], epoch: u64) -> Result<Snapshot, String> {
        let mut out = Snapshot::default();
        let mut error = [0 as c_char; 1024];
        if unsafe {
            (self.reset)(
                self.context.as_ptr(),
                &job,
                history.as_ptr(),
                history.len() as u64,
                epoch,
                &mut out,
                error.as_mut_ptr(),
                1024,
            )
        } != 0
        {
            return Err(buffer_string(&error));
        }
        Ok(out)
    }
    fn begin(&mut self, depth: u32) -> Result<Snapshot, String> {
        let mut out = Snapshot::default();
        let mut error = [0 as c_char; 1024];
        if unsafe {
            (self.begin)(
                self.context.as_ptr(),
                depth,
                &mut out,
                error.as_mut_ptr(),
                1024,
            )
        } != 0
        {
            return Err(buffer_string(&error));
        }
        Ok(out)
    }
    fn advance(&mut self, ns: u64) -> Result<(Snapshot, Stats), String> {
        let mut out = Snapshot::default();
        let mut stats = Stats::default();
        let mut error = [0 as c_char; 1024];
        if unsafe {
            (self.advance)(
                self.context.as_ptr(),
                ns,
                &mut out,
                std::ptr::null_mut(),
                0,
                &mut stats,
                error.as_mut_ptr(),
                1024,
            )
        } != 0
        {
            return Err(buffer_string(&error));
        }
        Ok((out, stats))
    }
    fn cancellation(&self) -> Arc<dyn Fn() + Send + Sync> {
        let ptr = self.context.as_ptr() as usize;
        let cancel = self.cancel;
        Arc::new(move || unsafe { cancel(ptr as *mut c_void) })
    }
}
impl Drop for Native {
    fn drop(&mut self) {
        if unsafe { (self.destroy)(self.context.as_ptr()) } != 0 {
            eprintln!("root pool retirement failed; retiring process");
            std::process::abort();
        }
        let _ = &self.library;
    }
}
pub struct HipRootPool {
    native: Native,
    evaluator: u32,
    quiescence: bool,
    stop: Arc<AtomicBool>,
    epoch: u64,
    telemetry: BatchTelemetry,
    stats: serde_json::Value,
}
impl HipRootPool {
    pub fn from_environment(evaluator: &str, quiescence: bool) -> Result<Self, String> {
        let evaluator = match evaluator {
            "material" => 0,
            "tapered" => 1,
            _ => return Err("hip-root-pool supports material/tapered".into()),
        };
        let path = std::env::var_os("CHEN_ROOT_POOL_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| "gpu/build/libchen_root_pool.so".into());
        Ok(Self {
            native: Native::new(&path)?,
            evaluator,
            quiescence,
            stop: Arc::new(AtomicBool::new(false)),
            epoch: 0,
            telemetry: BatchTelemetry::default(),
            stats: serde_json::Value::Null,
        })
    }
    fn checked(
        snapshot: &Snapshot,
        root_hash: u64,
        epoch: u64,
        node_limit: u64,
    ) -> Result<(), String> {
        let p = &snapshot.published;
        if snapshot.abi_version != 3
            || snapshot.struct_size != 408
            || snapshot.root_hash != root_hash
            || snapshot.root_epoch != epoch
            || snapshot.nodes > node_limit
            || snapshot.qnodes > snapshot.nodes
            || snapshot.root_count > 256
            || snapshot.completed_children > snapshot.root_count
            || snapshot.exact_children + snapshot.upper_children > snapshot.completed_children
            || snapshot.paused_children + snapshot.completed_children > snapshot.root_count
            || snapshot.abort_reason > 7
            || snapshot.reserved != 0
            || snapshot.padding != 0
            || p.abi_version != 1
            || p.struct_size != 304
            || p.reserved != 0
            || p.pv_length > 64
            || !matches!(p.kind, 0 | 3)
            || (p.kind == 3 && (p.score != 0 || p.completed_depth != 0 || p.pv_length != 0))
            || (p.kind == 0
                && (p.abort_reason != 0 || p.completed_depth > snapshot.requested_depth))
        {
            return Err("invalid root pool identity/publication/counter packet".into());
        }
        Ok(())
    }
    fn publish(snapshot: &Snapshot, result: &mut SearchResult) -> Result<bool, String> {
        result.nodes = snapshot.nodes;
        result.qnodes = snapshot.qnodes;
        let p = &snapshot.published;
        if p.kind != 0 || p.completed_depth <= u32::from(result.depth) {
            return Ok(false);
        }
        let pv = p.pv[..p.pv_length as usize]
            .iter()
            .copied()
            .map(Move::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        if pv.is_empty() {
            return Err("completed nonterminal root pool missing PV".into());
        }
        result.score = p.score;
        result.depth = p.completed_depth as u8;
        result.best_move = pv.first().copied();
        result.pv = pv;
        Ok(true)
    }
    fn iterations(
        &mut self,
        limits: &SearchLimits,
        root_hash: u64,
        node_limit: u64,
        started: Instant,
        mut result: SearchResult,
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        let mut epochs = Vec::new();
        let mut stop_reason = "depth";
        'depths: for depth in 1..=u32::from(limits.depth.clamp(1, 16)) {
            if result.nodes >= node_limit {
                stop_reason = "nodes";
                result.stopped = true;
                break;
            }
            if self.stop.load(Ordering::Relaxed) {
                stop_reason = "cancelled";
                result.stopped = true;
                break;
            }
            if limits.time.is_some_and(|time| started.elapsed() >= time) {
                stop_reason = "deadline";
                result.stopped = true;
                break;
            }
            let initial = self.native.begin(depth)?;
            Self::checked(&initial, root_hash, self.epoch, node_limit)?;
            loop {
                if self.stop.load(Ordering::Relaxed) {
                    stop_reason = "cancelled";
                    result.stopped = true;
                    break 'depths;
                }
                let ns = match limits.time {
                    Some(time) => {
                        let left = time.saturating_sub(started.elapsed());
                        if left.is_zero() {
                            stop_reason = "deadline";
                            result.stopped = true;
                            break 'depths;
                        }
                        left.as_nanos().min(20_000_000) as u64
                    }
                    None => 20_000_000,
                };
                let tick = Instant::now();
                let (s, stats) = self.native.advance(ns)?;
                Self::checked(&s, root_hash, self.epoch, node_limit)?;
                if !stats.kernel_ms.is_finite()
                    || !stats.d2h_ms.is_finite()
                    || stats.wall_clock_khz == 0
                {
                    return Err("invalid root pool native telemetry".into());
                }
                self.telemetry.accumulate(BatchTelemetry {
                    positions: s.nodes.saturating_sub(result.nodes),
                    h2d_ms: stats.h2d_ms,
                    kernel_ms: stats.kernel_ms,
                    d2h_ms: stats.d2h_ms,
                });
                let published = Self::publish(&s, &mut result)?;
                result.elapsed = started.elapsed();
                epochs.push(serde_json::json!({"depth":depth,"nodes":s.nodes,"qnodes":s.qnodes,"iteration_nodes":s.iteration_nodes,"completed_children":s.completed_children,"exact_children":s.exact_children,"upper_children":s.upper_children,"paused_children":s.paused_children,"abort_reason":s.abort_reason,"iteration_complete":s.iteration_complete,"elapsed_ticks":s.elapsed_ticks,"wall_ms":tick.elapsed().as_secs_f64()*1000.,"kernel_ms":stats.kernel_ms,"clock_khz":stats.wall_clock_khz,"scratch_bytes":stats.scratch_bytes}));
                if published {
                    report(&result);
                }
                if s.abort_reason != 0 {
                    stop_reason = match s.abort_reason {
                        1 => "nodes",
                        2 => "qply",
                        3 => "stack",
                        7 => "cancelled",
                        _ => {
                            return Err(format!(
                                "root pool rules/position failure {}",
                                s.abort_reason
                            ));
                        }
                    };
                    result.stopped = true;
                    break 'depths;
                }
                if s.iteration_complete != 0 {
                    if !published {
                        return Err(
                            "completed pool iteration failed to publish requested depth".into()
                        );
                    }
                    break;
                }
            }
            if result.score.abs() >= crate::search::MATE - 64 {
                stop_reason = "mate";
                break;
            }
        }
        result.elapsed = started.elapsed();
        self.stats = serde_json::json!({"abi":3,"bootstrap_nodes":1,"node_limit":node_limit,"quiescence":self.quiescence,"root_epoch":self.epoch,"completed_depth":result.depth,"stop_reason":stop_reason,"ordering":"previous completed PV first; canonical DFS","tt":false,"pvs":false,"epoch_ns":20000000,"worker_mapping":"one scalar DFS per root-child workgroup","epochs":epochs});
        Ok(result)
    }
}
impl SearchBackend for HipRootPool {
    fn name(&self) -> &str {
        "hip-root-pool"
    }
    fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    fn clear(&mut self) {
        self.telemetry = BatchTelemetry::default();
        self.stats = serde_json::Value::Null;
    }
    fn root_pool_stats(&self) -> Option<String> {
        Some(self.stats.to_string())
    }
    fn batch_telemetry(&self) -> BatchTelemetry {
        self.telemetry
    }
    fn search(
        &mut self,
        p: &mut Position,
        limits: SearchLimits,
        history: &[u64],
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        if history.len() > 65535 {
            return Err("pool history exceeds65535 hashes".into());
        }
        self.clear();
        let started = Instant::now();
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or("root pool epoch exhausted")?;
        let node_limit = limits.nodes.unwrap_or(1_000_000).min(1_000_000);
        let root = RuleState::from(&*p);
        let root_hash = root.hash;
        let job = Job {
            root,
            node_budget: node_limit,
            history_offset: 0,
            history_count: history.len() as u64,
            abi_version: 1,
            struct_size: 192,
            depth: 0,
            max_qply: 32,
            evaluator: self.evaluator,
            quiescence: self.quiescence as u32,
            root_ply: 0,
            reserved: 0,
            alpha: -32000,
            beta: 32000,
        };
        let bootstrap = self.native.reset(job, history, self.epoch)?;
        Self::checked(&bootstrap, root_hash, self.epoch, node_limit)?;
        if bootstrap.abort_reason != 0 {
            return Err(format!(
                "root pool admission failure {}",
                bootstrap.abort_reason
            ));
        }
        let fallback = if bootstrap.root_count > 0 {
            Some(Move::try_from(bootstrap.fallback)?)
        } else {
            None
        };
        let result = SearchResult {
            best_move: fallback,
            score: if bootstrap.terminal != 0 {
                bootstrap.published.score
            } else {
                0
            },
            depth: 0,
            nodes: 0,
            qnodes: 0,
            elapsed: started.elapsed(),
            pv: fallback.into_iter().collect(),
            stopped: false,
            hashfull: 0,
            book_move: None,
            selective: None,
        };
        if bootstrap.terminal != 0 {
            self.stats = serde_json::json!({"abi":3,"bootstrap_nodes":1,"stop_reason":"terminal","root_epoch":self.epoch});
            return Ok(result);
        }
        let done = Arc::new(AtomicBool::new(false));
        let stop = self.stop.clone();
        let cancel = self.native.cancellation();
        std::thread::scope(|scope| {
            struct Guard(Arc<AtomicBool>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            let _guard = Guard(done.clone());
            let monitor = done.clone();
            scope.spawn(move || {
                while !monitor.load(Ordering::Acquire) {
                    if stop.load(Ordering::Relaxed) {
                        cancel();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            let out = self.iterations(&limits, root_hash, node_limit, started, result, report);
            done.store(true, Ordering::Release);
            out
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_layout() {
        assert_eq!(std::mem::size_of::<Job>(), 192);
        assert_eq!(std::mem::size_of::<Packet>(), 304);
        assert_eq!(std::mem::size_of::<Snapshot>(), 408);
    }
    #[test]
    fn paused_retains_previous_publication() {
        let m = Move::try_from(RuleMove {
            from: 12,
            to: 28,
            promotion: 0,
            reserved: 0,
        })
        .unwrap();
        let mut result = SearchResult {
            best_move: Some(m),
            score: 42,
            depth: 2,
            nodes: 8,
            qnodes: 2,
            elapsed: Duration::ZERO,
            pv: vec![m],
            stopped: false,
            hashfull: 0,
            book_move: None,
            selective: None,
        };
        let mut s = Snapshot {
            nodes: 200,
            qnodes: 50,
            ..Snapshot::default()
        };
        s.published.kind = 3;
        s.published.score = 0;
        assert!(!HipRootPool::publish(&s, &mut result).unwrap());
        assert_eq!(result.depth, 2);
        assert_eq!(result.score, 42);
        assert_eq!(result.best_move, Some(m));
        assert_eq!(result.nodes, 200);
    }
    #[test]
    fn foreign_epoch_rejected() {
        let mut s = Snapshot {
            abi_version: 3,
            struct_size: 408,
            root_hash: 7,
            root_epoch: 3,
            ..Snapshot::default()
        };
        s.published.abi_version = 1;
        s.published.struct_size = 304;
        s.published.kind = 3;
        assert!(HipRootPool::checked(&s, 7, 3, 20).is_ok());
        assert!(HipRootPool::checked(&s, 7, 4, 20).is_err());
    }
}
