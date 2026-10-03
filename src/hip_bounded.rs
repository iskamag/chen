//! Experimental device DFS: host schedules complete depths, GPU owns rules,
//! evaluation, qsearch and bounds. Aborts never publish partial root values.
use crate::{
    Move, Position, SearchLimits, SearchResult,
    backend::{BatchTelemetry, BoundedIterationStats, BoundedSearchStats, SearchBackend},
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
const MAX_NODES: u64 = 1_000_000;
const MAX_DEVICE_NS: u64 = 5_000_000_000;
const EXACT: u32 = 0;
const ABORTED: u32 = 3;
const DEADLINE: u32 = 6;
const CANCELLED: u32 = 7;
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
#[derive(Clone, Copy, Debug)]
struct TimedJob {
    search: Job,
    elapsed_ns: u64,
    abi_version: u32,
    struct_size: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct TimedPacket {
    search: Packet,
    elapsed_ticks: u64,
    legal_root_count: u32,
    fallback: RuleMove,
    abi_version: u32,
    struct_size: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeStats {
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
    wall_clock_khz: u64,
    scratch_bytes: u64,
    history_capacity: u64,
}
type Create = unsafe extern "C" fn(i32, u32, *mut c_char, u64) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Cancel = unsafe extern "C" fn(*mut c_void);
type History =
    unsafe extern "C" fn(*mut c_void, *const u64, u64, *mut f64, *mut c_char, u64) -> i32;
type Run = unsafe extern "C" fn(
    *mut c_void,
    *const TimedJob,
    u32,
    *mut TimedPacket,
    *mut NativeStats,
    *mut c_char,
    u64,
) -> i32;
trait Runner: Send {
    fn history(&mut self, history: &[u64]) -> Result<f64, String>;
    fn clear_cancel(&mut self);
    fn cancellation(&self) -> Arc<dyn Fn() + Send + Sync>;
    fn run(&mut self, job: TimedJob) -> Result<(TimedPacket, NativeStats), String>;
}
struct NativeRunner {
    _library: Library,
    context: NonNull<c_void>,
    run: Run,
    history: History,
    cancel: Cancel,
    clear_cancel: Cancel,
    destroy: Destroy,
}
// SAFETY: unique context, serialized calls, with only ABI2's explicitly
// concurrent host atomic cancel callback overlapping a run. Library outlives all.
unsafe impl Send for NativeRunner {}
impl NativeRunner {
    fn new(path: &Path) -> Result<Self, String> {
        let library = Library::open(path)?;
        macro_rules! symbol {
            ($name:literal,$ty:ty) => {{
                // SAFETY: exact signatures and layouts from bounded_search_timed.h.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(library.symbol($name)?) }
            }};
        }
        let version = symbol!(
            c"chen_bounded_timed_abi_version",
            unsafe extern "C" fn() -> u32
        );
        if unsafe { version() } != 2 {
            return Err("expected bounded timed ABI2".into());
        }
        let create = symbol!(c"chen_bounded_timed_create", Create);
        let run = symbol!(c"chen_bounded_timed_run", Run);
        let history = symbol!(c"chen_bounded_timed_history", History);
        let cancel = symbol!(c"chen_bounded_timed_cancel", Cancel);
        let clear_cancel = symbol!(c"chen_bounded_timed_clear_cancel", Cancel);
        let destroy = symbol!(c"chen_bounded_timed_destroy", Destroy);
        let mut error = [0 as c_char; 1024];
        // SAFETY: owned error buffer, one persistent job slot, device0.
        let context = NonNull::new(unsafe { create(0, 1, error.as_mut_ptr(), error.len() as u64) })
            .ok_or_else(|| buffer_string(&error))?;
        Ok(Self {
            _library: library,
            context,
            run,
            history,
            cancel,
            clear_cancel,
            destroy,
        })
    }
}
impl Drop for NativeRunner {
    fn drop(&mut self) {
        // SAFETY: monitor joined before search returns, unique live context.
        // Uncertain retirement must not unload live kernel code or keep running.
        if unsafe { (self.destroy)(self.context.as_ptr()) } != 0 {
            eprintln!("bounded HIP cleanup/retirement failed; retiring process");
            std::process::abort();
        }
    }
}
impl Runner for NativeRunner {
    fn history(&mut self, history: &[u64]) -> Result<f64, String> {
        let mut ms = 0.;
        let mut error = [0 as c_char; 1024];
        // SAFETY: history is readable for its size; all ABI output buffers live.
        let status = unsafe {
            (self.history)(
                self.context.as_ptr(),
                history.as_ptr(),
                history.len() as u64,
                &mut ms,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        if status != 0 {
            return Err(buffer_string(&error));
        }
        Ok(ms)
    }
    fn clear_cancel(&mut self) {
        // SAFETY: serialized owner before monitor starts; no live kernel.
        unsafe { (self.clear_cancel)(self.context.as_ptr()) }
    }
    fn cancellation(&self) -> Arc<dyn Fn() + Send + Sync> {
        let pointer = self.context.as_ptr() as usize;
        let cancel = self.cancel;
        Arc::new(move || {
            // SAFETY: callback stays inside scoped search; ABI explicitly allows
            // concurrent cancel only. Context/library live until monitor joins.
            unsafe { cancel(pointer as *mut c_void) }
        })
    }
    fn run(&mut self, job: TimedJob) -> Result<(TimedPacket, NativeStats), String> {
        let mut packet = TimedPacket::default();
        let mut stats = NativeStats::default();
        let mut error = [0 as c_char; 1024];
        // SAFETY: complete correctly sized host packets and writable output.
        let status = unsafe {
            (self.run)(
                self.context.as_ptr(),
                &job,
                1,
                &mut packet,
                &mut stats,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        if status != 0 {
            return Err(buffer_string(&error));
        }
        Ok((packet, stats))
    }
}
pub struct HipBoundedSearch {
    runner: Box<dyn Runner>,
    evaluator: u32,
    quiescence: bool,
    stop: Arc<AtomicBool>,
    telemetry: BatchTelemetry,
    stats: BoundedSearchStats,
}
impl HipBoundedSearch {
    pub fn new(path: impl AsRef<Path>, evaluator: &str, quiescence: bool) -> Result<Self, String> {
        let id = match evaluator {
            "material" => 0,
            "tapered" => 1,
            _ => return Err("hip-bounded supports material or tapered".into()),
        };
        Ok(Self::with_runner(
            Box::new(NativeRunner::new(path.as_ref())?),
            id,
            quiescence,
        ))
    }
    pub fn from_environment(evaluator: &str, quiescence: bool) -> Result<Self, String> {
        let path = std::env::var_os("CHEN_BOUNDED_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| "gpu/build/libchen_bounded_timed.so".into());
        Self::new(path, evaluator, quiescence)
    }
    fn with_runner(runner: Box<dyn Runner>, evaluator: u32, quiescence: bool) -> Self {
        Self {
            runner,
            evaluator,
            quiescence,
            stop: Arc::new(AtomicBool::new(false)),
            telemetry: BatchTelemetry::default(),
            stats: BoundedSearchStats::default(),
        }
    }
    fn job(
        &self,
        p: &Position,
        history_count: usize,
        depth: u32,
        nodes: u64,
        ns: u64,
        quiescence: bool,
    ) -> TimedJob {
        TimedJob {
            search: Job {
                root: RuleState::from(p),
                node_budget: nodes,
                history_offset: 0,
                history_count: history_count as u64,
                abi_version: 1,
                struct_size: 192,
                depth,
                max_qply: 32,
                evaluator: self.evaluator,
                quiescence: u32::from(quiescence),
                root_ply: 0,
                reserved: 0,
                alpha: -32000,
                beta: 32000,
            },
            elapsed_ns: ns,
            abi_version: 2,
            struct_size: 208,
        }
    }
    fn checked_run(&mut self, job: TimedJob) -> Result<TimedPacket, String> {
        let (packet, stats) = self.runner.run(job)?;
        let p = &packet.search;
        if packet.abi_version != 2
            || packet.struct_size != 328
            || p.abi_version != 1
            || p.struct_size != 304
            || p.reserved != 0
            || p.kind > ABORTED
            || p.pv_length > 64
            || p.nodes > job.search.node_budget
            || p.qnodes > p.nodes
            || packet.legal_root_count > 256
            || !stats.h2d_ms.is_finite()
            || !stats.kernel_ms.is_finite()
            || !stats.d2h_ms.is_finite()
            || stats.wall_clock_khz == 0
            || (p.kind == ABORTED
                && (!(1..=7).contains(&p.abort_reason)
                    || p.score != 0
                    || p.completed_depth != 0
                    || p.pv_length != 0
                    || p.pv.iter().any(|m| *m != RuleMove::default())))
            || (p.kind != ABORTED && (p.abort_reason != 0 || p.completed_depth != job.search.depth))
        {
            return Err("invalid bounded timed result/telemetry packet".into());
        }
        self.telemetry.accumulate(BatchTelemetry {
            positions: p.nodes,
            h2d_ms: stats.h2d_ms,
            kernel_ms: stats.kernel_ms,
            d2h_ms: stats.d2h_ms,
        });
        self.stats.wall_clock_khz = stats.wall_clock_khz;
        self.stats.scratch_bytes = stats.scratch_bytes;
        Ok(packet)
    }
    fn search_iterations(
        &mut self,
        p: &Position,
        limits: &SearchLimits,
        history: &[u64],
        started: Instant,
        mut result: SearchResult,
        report: &mut dyn FnMut(&SearchResult),
    ) -> Result<SearchResult, String> {
        for depth in 1..=u32::from(limits.depth.clamp(1, 16)) {
            let remaining = self.stats.node_limit.saturating_sub(result.nodes);
            if self.stop.load(Ordering::Relaxed) {
                self.stats.stop_reason = "cancelled";
                result.stopped = true;
                break;
            }
            if remaining == 0 {
                self.stats.stop_reason = "nodes";
                result.stopped = true;
                break;
            }
            let allowance = match limits.time {
                Some(time) => {
                    let remaining = time.saturating_sub(started.elapsed());
                    if remaining.is_zero() {
                        self.stats.stop_reason = "deadline";
                        result.stopped = true;
                        break;
                    }
                    remaining.as_nanos().min(u128::from(MAX_DEVICE_NS)) as u64
                }
                None => MAX_DEVICE_NS,
            };
            let job = self.job(
                p,
                history.len(),
                depth,
                remaining,
                allowance,
                self.quiescence,
            );
            let iteration_started = Instant::now();
            let packet = self.checked_run(job)?;
            let native = &packet.search;
            result.nodes += native.nodes;
            result.qnodes += native.qnodes;
            self.stats.iterations.push(BoundedIterationStats {
                requested_depth: depth,
                kind: native.kind,
                abort_reason: native.abort_reason,
                nodes: native.nodes,
                qnodes: native.qnodes,
                elapsed_ticks: packet.elapsed_ticks,
                wall_ms: iteration_started.elapsed().as_secs_f64() * 1000.,
            });
            if native.kind == ABORTED {
                self.stats.stop_reason = match native.abort_reason {
                    1 => "nodes",
                    2 => "qply",
                    3 => "stack",
                    DEADLINE => "deadline",
                    CANCELLED => "cancelled",
                    _ => {
                        return Err(format!(
                            "bounded device rule/position failure {}",
                            native.abort_reason
                        ));
                    }
                };
                result.stopped = true;
                break;
            }
            if native.kind != EXACT {
                return Err("full-window bounded iteration returned a non-Exact bound".into());
            }
            let pv = native.pv[..native.pv_length as usize]
                .iter()
                .copied()
                .map(Move::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            result.score = native.score;
            result.depth = depth as u8;
            if let Some(mv) = pv.first() {
                result.best_move = Some(*mv);
                result.pv = pv;
            } else {
                result.pv = result.best_move.into_iter().collect();
            }
            self.stats.completed_depth = result.depth;
            result.elapsed = started.elapsed();
            report(&result);
            if result.score.abs() >= crate::search::MATE - 64 {
                self.stats.stop_reason = "mate";
                break;
            }
        }
        result.elapsed = started.elapsed();
        Ok(result)
    }
}
impl SearchBackend for HipBoundedSearch {
    fn name(&self) -> &str {
        "hip-bounded"
    }
    fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    fn clear(&mut self) {
        self.telemetry = BatchTelemetry::default();
        self.stats = BoundedSearchStats::default();
    }
    fn bounded_stats(&self) -> Option<BoundedSearchStats> {
        Some(self.stats.clone())
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
            return Err("bounded history exceeds65535 prior hashes".into());
        }
        let started = Instant::now();
        self.clear();
        self.runner.clear_cancel();
        self.stats.node_limit = limits.nodes.unwrap_or(MAX_NODES).min(MAX_NODES);
        self.stats.quiescence = self.quiescence;
        self.stats.stop_reason = "depth";
        self.stats.history_upload_ms = self.runner.history(history)?;
        self.telemetry.h2d_ms += self.stats.history_upload_ms;
        // Mandatory GPU root admission supplies legal fallback even for nodes0 or
        // time0. One bootstrap node is separate from the requested search budget.
        let bootstrap = self.checked_run(self.job(p, history.len(), 0, 1, 50_000_000, false))?;
        if bootstrap.search.kind != EXACT {
            return Err("bounded GPU root bootstrap did not complete".into());
        }
        self.stats.bootstrap_nodes = bootstrap.search.nodes;
        let fallback = if bootstrap.legal_root_count > 0 {
            Some(Move::try_from(bootstrap.fallback)?)
        } else {
            None
        };
        let result = SearchResult {
            best_move: fallback,
            score: if fallback.is_none() {
                bootstrap.search.score
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
            selective: None,
        };
        if fallback.is_none() {
            self.stats.stop_reason = "terminal";
            return Ok(result);
        }
        let done = Arc::new(AtomicBool::new(false));
        let stop = self.stop.clone();
        let cancel = self.runner.cancellation();
        // Monitor issues only a native coherent atomic cancellation store, never
        // HIP work or CPU chess evaluation. It joins before any context destruction.
        let mut answer = std::thread::scope(|scope| {
            struct MonitorGuard(Arc<AtomicBool>);
            impl Drop for MonitorGuard {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            let _guard = MonitorGuard(done.clone());
            let monitor_done = done.clone();
            scope.spawn(move || {
                while !monitor_done.load(Ordering::Acquire) {
                    if stop.load(Ordering::Relaxed) {
                        cancel();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            let answer = self.search_iterations(p, &limits, history, started, result, report);
            done.store(true, Ordering::Release);
            answer
        })?;
        answer.elapsed = started.elapsed();
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::VecDeque, sync::Mutex};
    struct Mock {
        answers: VecDeque<TimedPacket>,
        jobs: Arc<Mutex<Vec<TimedJob>>>,
        history: Arc<Mutex<Vec<u64>>>,
        cancel: Arc<AtomicBool>,
        wait_last: bool,
    }
    impl Runner for Mock {
        fn history(&mut self, h: &[u64]) -> Result<f64, String> {
            *self.history.lock().unwrap() = h.to_vec();
            Ok(0.)
        }
        fn clear_cancel(&mut self) {
            self.cancel.store(false, Ordering::SeqCst);
        }
        fn cancellation(&self) -> Arc<dyn Fn() + Send + Sync> {
            let cancel = self.cancel.clone();
            Arc::new(move || cancel.store(true, Ordering::SeqCst))
        }
        fn run(&mut self, j: TimedJob) -> Result<(TimedPacket, NativeStats), String> {
            self.jobs.lock().unwrap().push(j);
            if self.wait_last && self.answers.len() == 1 {
                let start = Instant::now();
                while !self.cancel.load(Ordering::SeqCst)
                    && start.elapsed() < Duration::from_secs(1)
                {
                    std::thread::sleep(Duration::from_millis(1));
                }
                assert!(self.cancel.load(Ordering::SeqCst));
            }
            Ok((
                self.answers.pop_front().expect("undeclared mock call"),
                NativeStats {
                    wall_clock_khz: 100000,
                    ..Default::default()
                },
            ))
        }
    }
    fn result(depth: u32, nodes: u64, best: Option<Move>) -> TimedPacket {
        let mut p = TimedPacket {
            abi_version: 2,
            struct_size: 328,
            legal_root_count: 20,
            fallback: RuleMove::try_from(Move::new(8, 16, None)).unwrap(),
            ..Default::default()
        };
        p.search = Packet {
            abi_version: 1,
            struct_size: 304,
            completed_depth: depth,
            nodes,
            ..Default::default()
        };
        if let Some(m) = best {
            p.search.pv_length = 1;
            p.search.pv[0] = RuleMove::try_from(m).unwrap();
        }
        p
    }
    fn abort(reason: u32, nodes: u64) -> TimedPacket {
        let mut p = result(0, nodes, None);
        p.search.kind = ABORTED;
        p.search.abort_reason = reason;
        p
    }
    type MockBackend = (
        HipBoundedSearch,
        Arc<Mutex<Vec<TimedJob>>>,
        Arc<Mutex<Vec<u64>>>,
    );

    fn backend(answers: Vec<TimedPacket>, wait_last: bool) -> MockBackend {
        let jobs = Arc::new(Mutex::new(Vec::new()));
        let history = Arc::new(Mutex::new(Vec::new()));
        (
            HipBoundedSearch::with_runner(
                Box::new(Mock {
                    answers: answers.into(),
                    jobs: jobs.clone(),
                    history: history.clone(),
                    cancel: Arc::new(AtomicBool::new(false)),
                    wait_last,
                }),
                1,
                true,
            ),
            jobs,
            history,
        )
    }
    #[test]
    fn ffi_layout_and_no_missing_library_fallback() {
        assert_eq!(std::mem::size_of::<Job>(), 192);
        assert_eq!(std::mem::size_of::<Packet>(), 304);
        assert_eq!(std::mem::size_of::<TimedJob>(), 208);
        assert_eq!(std::mem::size_of::<TimedPacket>(), 328);
        assert_eq!(std::mem::size_of::<NativeStats>(), 48);
        assert!(HipBoundedSearch::new("/nonexistent/chen.so", "tapered", true).is_err());
    }
    #[test]
    fn abort_preserves_exact_root_and_counts_attempted_work() {
        let chosen = Move::new(12, 28, None);
        let (mut b, jobs, h) = backend(
            vec![result(0, 1, None), result(1, 5, Some(chosen)), abort(1, 7)],
            false,
        );
        let mut p = Position::startpos();
        let original = p;
        let mut reported = Vec::new();
        let r = b
            .search(
                &mut p,
                SearchLimits {
                    depth: 4,
                    nodes: Some(12),
                    time: None,
                },
                &[4, 5],
                &mut |r| reported.push(r.clone()),
            )
            .unwrap();
        assert_eq!(r.best_move, Some(chosen));
        assert_eq!(r.pv, vec![chosen]);
        assert_eq!(r.depth, 1);
        assert_eq!(r.nodes, 12);
        assert!(r.stopped);
        assert_eq!(reported.len(), 1);
        assert_eq!(p, original);
        assert_eq!(*h.lock().unwrap(), vec![4, 5]);
        let j = jobs.lock().unwrap();
        assert_eq!(j.len(), 3);
        assert_eq!(j[1].search.history_count, 2);
        assert_eq!(j[1].search.root_ply, 0);
        assert_eq!(j[2].search.node_budget, 7);
        assert_eq!(b.stats.completed_depth, 1);
    }
    #[test]
    fn zero_budget_and_zero_time_keep_gpu_fallback_without_search_claim() {
        for limits in [
            SearchLimits {
                nodes: Some(0),
                ..Default::default()
            },
            SearchLimits {
                time: Some(Duration::ZERO),
                ..Default::default()
            },
        ] {
            let (mut b, jobs, _) = backend(vec![result(0, 1, None)], false);
            let r = b
                .search(&mut Position::startpos(), limits, &[], &mut |_| {
                    panic!("no completed iteration")
                })
                .unwrap();
            assert_eq!(r.depth, 0);
            assert_eq!(r.nodes, 0);
            assert_eq!(r.score, 0);
            assert!(r.stopped);
            assert_eq!(r.best_move, Some(Move::new(8, 16, None)));
            assert_eq!(jobs.lock().unwrap().len(), 1);
        }
    }
    #[test]
    fn malformed_abort_and_nonexact_full_window_are_rejected() {
        let mut bad = abort(DEADLINE, 2);
        bad.search.score = 42;
        for packet in [bad, {
            let mut p = result(1, 2, None);
            p.search.kind = 1;
            p
        }] {
            let (mut b, _, _) = backend(vec![result(0, 1, None), packet], false);
            assert!(
                b.search(
                    &mut Position::startpos(),
                    SearchLimits {
                        depth: 1,
                        ..Default::default()
                    },
                    &[],
                    &mut |_| {}
                )
                .is_err()
            );
        }
    }
    #[test]
    fn concurrent_stop_uses_cancel_and_retains_completed_depth() {
        let chosen = Move::new(12, 28, None);
        let (mut b, _, _) = backend(
            vec![
                result(0, 1, None),
                result(1, 5, Some(chosen)),
                abort(CANCELLED, 3),
            ],
            true,
        );
        let stop = b.stop_handle();
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(15));
            stop.store(true, Ordering::Relaxed);
        });
        let r = b
            .search(
                &mut Position::startpos(),
                SearchLimits {
                    depth: 8,
                    ..Default::default()
                },
                &[],
                &mut |_| {},
            )
            .unwrap();
        setter.join().unwrap();
        assert_eq!(r.depth, 1);
        assert_eq!(r.best_move, Some(chosen));
        assert_eq!(r.nodes, 8);
        assert_eq!(b.stats.stop_reason, "cancelled");
    }
    #[test]
    fn terminal_bootstrap_needs_no_iteration_or_cpu_legality() {
        let mut terminal = result(0, 1, None);
        terminal.legal_root_count = 0;
        terminal.search.score = -30000;
        let (mut b, jobs, _) = backend(vec![terminal], false);
        let r = b
            .search(
                &mut Position::startpos(),
                SearchLimits::default(),
                &[],
                &mut |_| {},
            )
            .unwrap();
        assert!(r.best_move.is_none());
        assert_eq!(r.score, -30000);
        assert!(!r.stopped);
        assert_eq!(b.stats.stop_reason, "terminal");
        assert_eq!(jobs.lock().unwrap().len(), 1);
    }
}
