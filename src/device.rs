//! Safe ownership of the versioned, synchronous GPU rules/tree ABI.
//! The library stays loaded until after the uniquely owned context is destroyed.
use crate::{
    Move,
    hip::{Library, buffer_string},
    resident::{ResidentTree, TreeError, TreeSpan, TreeStats},
    rules::{LegalBatch, RuleMove, RuleState, RulesBackend},
};
use std::{
    ffi::{c_char, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
};

#[repr(C)]
#[derive(Default)]
struct Timings {
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
    count: u64,
    capacity: u64,
}
#[repr(C)]
#[derive(Default)]
struct ResultPacket {
    score: i32,
    pv_length: u32,
}
type Create = unsafe extern "C" fn(i32, u64, *mut c_char, u64) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Legal = unsafe extern "C" fn(
    *mut c_void,
    *const RuleState,
    u64,
    *mut RuleMove,
    *mut u32,
    *mut u32,
    *mut Timings,
    *mut c_char,
    u64,
) -> i32;
type Make = unsafe extern "C" fn(
    *mut c_void,
    *const RuleState,
    *const RuleMove,
    u64,
    *mut RuleState,
    *mut Timings,
    *mut c_char,
    u64,
) -> i32;
type Reset =
    unsafe extern "C" fn(*mut c_void, *const RuleState, *const u64, u64, *mut c_char, u64) -> i32;
type Stage =
    unsafe extern "C" fn(*mut c_void, u64, u64, u32, *mut TreeStats, *mut c_char, u64) -> i32;
type Reduce = unsafe extern "C" fn(*mut c_void, u64, u64, *mut TreeStats, *mut c_char, u64) -> i32;
type Finish = unsafe extern "C" fn(
    *mut c_void,
    *mut ResultPacket,
    *mut RuleMove,
    u64,
    *mut TreeStats,
    *mut c_char,
    u64,
) -> i32;
type Download =
    unsafe extern "C" fn(*mut c_void, u64, u64, *mut RuleState, *mut c_char, u64) -> i32;
type Perft = unsafe extern "C" fn(
    *mut c_void,
    *const RuleState,
    u32,
    *mut u64,
    *mut TreeStats,
    *mut c_char,
    u64,
) -> i32;

pub struct HipRules {
    _library: Library,
    context: NonNull<c_void>,
    destroy: Destroy,
    legal: Legal,
    make: Make,
    reset: Reset,
    classify: Stage,
    expand: Stage,
    evaluate: Stage,
    reduce: Reduce,
    finish: Finish,
    download: Download,
    perft: Perft,
    capacity: usize,
}
// SAFETY: exclusive context ownership; all operations require &mut self. The C
// ABI sets the device on every call and permits serialized host-thread transfer.
unsafe impl Send for HipRules {}
fn status(code: i32, error: &[c_char]) -> Result<(), TreeError> {
    match code {
        0 => Ok(()),
        -2 => Err(TreeError::Capacity(buffer_string(error))),
        _ => Err(TreeError::Backend(buffer_string(error))),
    }
}
impl HipRules {
    pub fn new(path: impl AsRef<Path>, capacity: usize) -> Result<Self, String> {
        Self::new_bound(path.as_ref(), capacity, None)
    }
    pub(crate) fn new_with_runtime(
        path: &Path,
        capacity: usize,
        identity: [u64; 5],
    ) -> Result<Self, String> {
        Self::new_bound(path, capacity, Some(identity))
    }
    fn new_bound(path: &Path, capacity: usize, identity: Option<[u64; 5]>) -> Result<Self, String> {
        if capacity == 0 {
            return Err("GPU tree capacity must be positive".into());
        }
        let library = Library::open(path)?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                // SAFETY: signatures exactly match gpu/chen_rules.h ABI 1.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(library.symbol($name)?) }
            }};
        }
        let abi = symbol!(c"chen_rules_abi_version", unsafe extern "C" fn() -> u32);
        // SAFETY: no-argument version query; no context functions used before check.
        if unsafe { abi() } != 1 {
            return Err("unsupported Chen rules ABI (expected 1)".into());
        }
        if let Some(expected) = identity {
            let token = symbol!(
                c"chen_replay_runtime_identity",
                unsafe extern "C" fn(u32) -> u64
            );
            let external = symbol!(c"chen_external_abi_version", unsafe extern "C" fn() -> u32);
            // SAFETY: metadata queries, before context creation or pointer exchange.
            if unsafe { external() } != 1 {
                return Err("unsupported external evaluator ABI".into());
            }
            for (i, expected_token) in expected.into_iter().enumerate() {
                if expected_token == 0 || unsafe { token(i as u32) } != expected_token {
                    return Err(format!(
                        "neural/rules HIP runtime identity mismatch at selector {i}"
                    ));
                }
            }
        }
        let create = symbol!(c"chen_rules_create", Create);
        let destroy = symbol!(c"chen_rules_destroy", Destroy);
        let legal = symbol!(c"chen_rules_legal", Legal);
        let make = symbol!(c"chen_rules_make", Make);
        let reset = symbol!(c"chen_rules_reset", Reset);
        let classify = symbol!(c"chen_rules_classify", Stage);
        let expand = symbol!(c"chen_rules_expand", Stage);
        let evaluate = symbol!(c"chen_rules_evaluate", Stage);
        let reduce = symbol!(c"chen_rules_reduce", Reduce);
        let finish = symbol!(c"chen_rules_finish", Finish);
        let download = symbol!(c"chen_rules_download", Download);
        let perft = symbol!(c"chen_rules_perft", Perft);
        let mut error = [0 as c_char; 512];
        // SAFETY: writable error buffer and checked capacity; returned pointer is owned.
        let context = NonNull::new(unsafe {
            create(0, capacity as u64, error.as_mut_ptr(), error.len() as u64)
        })
        .ok_or_else(|| buffer_string(&error))?;
        Ok(Self {
            _library: library,
            context,
            destroy,
            legal,
            make,
            reset,
            classify,
            expand,
            evaluate,
            reduce,
            finish,
            download,
            perft,
            capacity,
        })
    }
    pub fn from_environment(capacity: usize) -> Result<Self, String> {
        Self::new(
            std::env::var_os("CHEN_GPU_LIBRARY")
                .map(PathBuf::from)
                .unwrap_or_else(|| "gpu/build/libchen_gpu.so".into()),
            capacity,
        )
    }
    pub(crate) fn evaluate_external(
        &mut self,
        span: TreeSpan,
        descriptor: &crate::neural::ExternalEvaluator,
    ) -> Result<TreeStats, TreeError> {
        type External = unsafe extern "C" fn(
            *mut c_void,
            u64,
            u64,
            *const crate::neural::ExternalEvaluator,
            *mut TreeStats,
            *mut c_char,
            u64,
        ) -> i32;
        let ptr = self
            ._library
            .symbol(c"chen_rules_evaluate_external")
            .map_err(TreeError::Backend)?;
        // SAFETY: exact independently versioned chen_external.h ABI.
        let evaluate = unsafe { std::mem::transmute::<*mut c_void, External>(ptr) };
        let mut stats = TreeStats::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: exclusively owned live context, borrowed descriptor and owner
        // live throughout synchronous call, valid stats/error output buffers.
        status(
            unsafe {
                evaluate(
                    self.context.as_ptr(),
                    span.start,
                    span.count,
                    descriptor,
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )?;
        Ok(stats)
    }
    pub fn perft(&mut self, root: &RuleState, depth: u32) -> Result<(u64, TreeStats), TreeError> {
        let mut nodes = 0;
        let mut stats = TreeStats::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: live exclusive context and properly sized input/output ABI packets.
        status(
            unsafe {
                (self.perft)(
                    self.context.as_ptr(),
                    root,
                    depth,
                    &mut nodes,
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )?;
        Ok((nodes, stats))
    }
    /// Debug/oracle boundary; normal resident search never downloads arena states.
    pub fn download(&mut self, span: TreeSpan) -> Result<Vec<RuleState>, TreeError> {
        let count = usize::try_from(span.count)
            .map_err(|_| TreeError::Capacity("download size overflow".into()))?;
        if count > self.capacity {
            return Err(TreeError::Capacity("download exceeds capacity".into()));
        }
        // All-zero scalar/array state is valid Rust data; C overwrites before use.
        let empty = RuleState {
            pieces: [0; 12],
            side: 0,
            castling: 0,
            ep_square: -1,
            halfmove: 0,
            fullmove: 1,
            reserved: 0,
            hash: 0,
        };
        let mut out = vec![empty; count];
        let mut error = [0 as c_char; 512];
        // SAFETY: output has advertised count; C validates span before writing.
        status(
            unsafe {
                (self.download)(
                    self.context.as_ptr(),
                    span.start,
                    span.count,
                    out.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )?;
        Ok(out)
    }
    fn stage(&mut self, call: Stage, span: TreeSpan, option: u32) -> Result<TreeStats, TreeError> {
        let mut stats = TreeStats::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: exclusive context and ABI outputs; C validates range/option.
        status(
            unsafe {
                call(
                    self.context.as_ptr(),
                    span.start,
                    span.count,
                    option,
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )?;
        Ok(stats)
    }
}
impl Drop for HipRules {
    fn drop(&mut self) {
        // SAFETY: unique context is destroyed once, before Library fields drop.
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}
impl RulesBackend for HipRules {
    fn name(&self) -> &str {
        "hip-rules"
    }
    fn legal_batch(&mut self, states: &[RuleState]) -> Result<LegalBatch, String> {
        if states.len() > self.capacity {
            return Err("rules batch exceeds GPU capacity".into());
        }
        let size = states
            .len()
            .checked_mul(256)
            .ok_or("move buffer size overflow")?;
        let mut moves = vec![RuleMove::default(); size];
        let mut counts = vec![0; states.len()];
        let mut flags = vec![0; states.len()];
        let mut timings = Timings::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: all arrays match header's count/count*256 requirements.
        status(
            unsafe {
                (self.legal)(
                    self.context.as_ptr(),
                    states.as_ptr(),
                    states.len() as u64,
                    moves.as_mut_ptr(),
                    counts.as_mut_ptr(),
                    flags.as_mut_ptr(),
                    &mut timings,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )
        .map_err(|e| e.to_string())?;
        let mut legal = Vec::with_capacity(states.len());
        for (i, &count) in counts.iter().enumerate() {
            if count > 256 || flags[i] & !3 != 0 {
                return Err("invalid GPU legal result".into());
            }
            legal.push(
                moves[i * 256..i * 256 + count as usize]
                    .iter()
                    .copied()
                    .map(Move::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(LegalBatch {
            moves: legal,
            in_check: flags.iter().map(|f| f & 1 != 0).collect(),
            insufficient: flags.iter().map(|f| f & 2 != 0).collect(),
        })
    }
    fn make_batch(
        &mut self,
        states: &[RuleState],
        moves: &[Move],
    ) -> Result<Vec<RuleState>, String> {
        if states.len() != moves.len() || states.len() > self.capacity {
            return Err("invalid GPU transition batch length".into());
        }
        let packed = moves
            .iter()
            .copied()
            .map(RuleMove::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = states.to_vec();
        let mut timings = Timings::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: input/output arrays have matching advertised lengths.
        status(
            unsafe {
                (self.make)(
                    self.context.as_ptr(),
                    states.as_ptr(),
                    packed.as_ptr(),
                    states.len() as u64,
                    out.as_mut_ptr(),
                    &mut timings,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )
        .map_err(|e| e.to_string())?;
        Ok(out)
    }
}
impl ResidentTree for HipRules {
    fn name(&self) -> &str {
        "hip-resident"
    }
    fn reset(&mut self, root: RuleState, history: &[u64]) -> Result<(), TreeError> {
        let mut error = [0 as c_char; 512];
        // SAFETY: root/history remain live throughout the synchronous call.
        status(
            unsafe {
                (self.reset)(
                    self.context.as_ptr(),
                    &root,
                    history.as_ptr(),
                    history.len() as u64,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )
    }
    fn classify(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError> {
        self.stage(self.classify, span, u32::from(adjudicate))
    }
    fn expand(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError> {
        self.stage(self.expand, span, u32::from(adjudicate))
    }
    fn evaluate(&mut self, span: TreeSpan, evaluator: u32) -> Result<TreeStats, TreeError> {
        self.stage(self.evaluate, span, evaluator)
    }
    fn reduce(&mut self, span: TreeSpan) -> Result<TreeStats, TreeError> {
        let mut stats = TreeStats::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: writable ABI outputs, C checks submitted span.
        status(
            unsafe {
                (self.reduce)(
                    self.context.as_ptr(),
                    span.start,
                    span.count,
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )?;
        Ok(stats)
    }
    fn finish(&mut self) -> Result<(i32, Vec<Move>, TreeStats), TreeError> {
        let mut result = ResultPacket::default();
        let mut stats = TreeStats::default();
        // A simple path cannot contain more edges than retained arena positions.
        let mut pv = vec![RuleMove::default(); self.capacity];
        let mut error = [0 as c_char; 512];
        // SAFETY: outputs sized exactly as advertised and context is exclusively held.
        status(
            unsafe {
                (self.finish)(
                    self.context.as_ptr(),
                    &mut result,
                    pv.as_mut_ptr(),
                    pv.len() as u64,
                    &mut stats,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            },
            &error,
        )?;
        if result.pv_length as usize > pv.len() {
            return Err(TreeError::Backend("invalid GPU PV length".into()));
        }
        let moves = pv[..result.pv_length as usize]
            .iter()
            .copied()
            .map(Move::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((result.score, moves, stats))
    }
}

// The selective ABI owns a separate private arena. It deliberately exposes no
// raw context handle and cannot be substituted for the full-width tree API.
type SelectiveReset = unsafe extern "C" fn(
    *mut c_void,
    *const RuleState,
    *const u64,
    u64,
    *const crate::selective::SelectiveConfig,
    *mut crate::selective::SelectiveStats,
    *mut c_char,
    u64,
) -> i32;
type SelectiveAdvance = unsafe extern "C" fn(
    *mut c_void,
    u64,
    u32,
    *const crate::neural::ExternalEvaluator,
    *mut crate::selective::SelectiveStats,
    *mut c_char,
    u64,
) -> i32;
type SelectiveFinish = unsafe extern "C" fn(
    *mut c_void,
    *mut ResultPacket,
    *mut RuleMove,
    u64,
    *mut crate::selective::SelectiveStats,
    *mut c_char,
    u64,
) -> i32;

pub struct HipSelectiveTree {
    _library: Library,
    context: NonNull<c_void>,
    destroy: Destroy,
    reset: SelectiveReset,
    advance: SelectiveAdvance,
    finish: SelectiveFinish,
    capacity: usize,
    poisoned: bool,
}
// SAFETY: uniquely owned context, exclusively serialized operations; every
// native device call establishes the device before touching allocations.
unsafe impl Send for HipSelectiveTree {}
impl HipSelectiveTree {
    pub fn new(path: impl AsRef<Path>, capacity: usize) -> Result<Self, String> {
        Self::new_bound(path.as_ref(), capacity, None)
    }
    pub(crate) fn new_with_runtime(
        path: &Path,
        capacity: usize,
        identity: [u64; 5],
    ) -> Result<Self, String> {
        Self::new_bound(path, capacity, Some(identity))
    }
    fn new_bound(path: &Path, capacity: usize, identity: Option<[u64; 5]>) -> Result<Self, String> {
        if capacity == 0 {
            return Err("selective GPU capacity must be positive".into());
        }
        let library = Library::open(path)?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                // SAFETY: signatures match independently versioned chen_selective.h.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(library.symbol($name)?) }
            }};
        }
        let abi = symbol!(c"chen_selective_abi_version", unsafe extern "C" fn() -> u32);
        // SAFETY: no-argument metadata query before context creation.
        if unsafe { abi() } != crate::selective::SELECTIVE_ABI {
            return Err("unsupported selective ABI (expected1)".into());
        }
        if let Some(expected) = identity {
            let token = symbol!(
                c"chen_replay_runtime_identity",
                unsafe extern "C" fn(u32) -> u64
            );
            let external = symbol!(c"chen_external_abi_version", unsafe extern "C" fn() -> u32);
            // SAFETY: metadata-only calls, no pointer exchange before verification.
            if unsafe { external() } != 1 {
                return Err("unsupported selective external evaluator ABI".into());
            }
            for (i, value) in expected.into_iter().enumerate() {
                if value == 0 || unsafe { token(i as u32) } != value {
                    return Err(format!(
                        "selective neural/rules runtime identity mismatch at selector{i}"
                    ));
                }
            }
        }
        let create = symbol!(c"chen_selective_create", Create);
        let destroy = symbol!(c"chen_selective_destroy", Destroy);
        let reset = symbol!(c"chen_selective_reset", SelectiveReset);
        let advance = symbol!(c"chen_selective_advance", SelectiveAdvance);
        let finish = symbol!(c"chen_selective_finish", SelectiveFinish);
        let mut error = [0 as c_char; 512];
        // SAFETY: checked capacity and writable bounded error buffer; owned pointer.
        let context = NonNull::new(unsafe {
            create(0, capacity as u64, error.as_mut_ptr(), error.len() as u64)
        })
        .ok_or_else(|| buffer_string(&error))?;
        Ok(Self {
            _library: library,
            context,
            destroy,
            reset,
            advance,
            finish,
            capacity,
            poisoned: true,
        })
    }
    pub fn from_environment(capacity: usize) -> Result<Self, String> {
        Self::new(
            std::env::var_os("CHEN_GPU_LIBRARY")
                .map(PathBuf::from)
                .unwrap_or_else(|| "gpu/build/libchen_gpu.so".into()),
            capacity,
        )
    }
    fn checked(
        &mut self,
        code: i32,
        error: &[c_char],
        stats: crate::selective::SelectiveStats,
    ) -> Result<crate::selective::SelectiveStats, TreeError> {
        let error = if code != 0 {
            Some(buffer_string(error))
        } else if stats.abi_version != crate::selective::SELECTIVE_ABI
            || stats.struct_size as usize != std::mem::size_of::<crate::selective::SelectiveStats>()
            || stats.reserved != 0
            || stats.arena_positions > self.capacity as u64
            || stats.classified_positions > stats.arena_positions
            || stats.evaluated_positions > stats.classified_positions
            || stats.open_frontier_leaves > stats.arena_positions
            || stats.minimum_frontier_depth > stats.maximum_frontier_depth
            || stats.maximum_frontier_depth > 64
            || stats.pv_depth > 64
            || stats.root_flags & !15 != 0
            || stats.stop_reason > 5
            || !stats.kernel_ms.is_finite()
            || stats.kernel_ms < 0.0
        {
            Some("malformed selective ABI stats; reset required".into())
        } else {
            None
        };
        if let Some(error) = error {
            self.poisoned = true;
            return Err(TreeError::Backend(error));
        }
        Ok(stats)
    }
    pub(crate) fn advance_external(
        &mut self,
        cap: u64,
        evaluator: u32,
        descriptor: Option<&crate::neural::ExternalEvaluator>,
    ) -> Result<crate::selective::SelectiveStats, TreeError> {
        if self.poisoned {
            return Err(TreeError::Backend(
                "selective GPU arena invalid; reset required".into(),
            ));
        }
        let mut stats = crate::selective::SelectiveStats::empty();
        let mut error = [0 as c_char; 512];
        // SAFETY: exclusively owned live context; borrowed descriptor's plugin
        // and owner are retained by wrapper throughout this synchronous call.
        let code = unsafe {
            (self.advance)(
                self.context.as_ptr(),
                cap,
                evaluator,
                descriptor.map_or(std::ptr::null(), |d| d),
                &mut stats,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        self.checked(code, &error, stats)
    }
}
impl crate::selective::SelectiveTree for HipSelectiveTree {
    fn name(&self) -> &str {
        "hip-selective"
    }
    fn reset(
        &mut self,
        root: RuleState,
        history: &[u64],
        config: crate::selective::SelectiveConfig,
    ) -> Result<crate::selective::SelectiveStats, TreeError> {
        self.poisoned = true;
        config.validate()?;
        if history.len() > u16::MAX as usize {
            return Err(TreeError::Backend("history exceeds ABI capacity".into()));
        }
        let mut stats = crate::selective::SelectiveStats::empty();
        let mut error = [0 as c_char; 512];
        // SAFETY: root/config/history are borrowed for synchronous upload only.
        let code = unsafe {
            (self.reset)(
                self.context.as_ptr(),
                &root,
                history.as_ptr(),
                history.len() as u64,
                &config,
                &mut stats,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        let stats = self.checked(code, &error, stats)?;
        self.poisoned = false;
        Ok(stats)
    }
    fn advance(
        &mut self,
        cap: u64,
        evaluator: u32,
    ) -> Result<crate::selective::SelectiveStats, TreeError> {
        self.advance_external(cap, evaluator, None)
    }
    fn finish(&mut self) -> Result<(i32, Vec<Move>, crate::selective::SelectiveStats), TreeError> {
        if self.poisoned {
            return Err(TreeError::Backend(
                "selective GPU arena invalid; reset required".into(),
            ));
        }
        let mut packet = ResultPacket::default();
        let mut moves = [RuleMove::default(); 64];
        let mut stats = crate::selective::SelectiveStats::empty();
        let mut error = [0 as c_char; 512];
        // SAFETY: writable output storage exactly spans supplied capacities.
        let code = unsafe {
            (self.finish)(
                self.context.as_ptr(),
                &mut packet,
                moves.as_mut_ptr(),
                moves.len() as u64,
                &mut stats,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        let stats = self.checked(code, &error, stats)?;
        if packet.pv_length as usize > moves.len()
            || !stats.published()
            || packet.score.unsigned_abs() > crate::search::MATE as u32
        {
            self.poisoned = true;
            return Err(TreeError::Backend(
                "malformed selective root/PV; reset required".into(),
            ));
        }
        let pv = moves[..packet.pv_length as usize]
            .iter()
            .copied()
            .map(Move::try_from)
            .collect::<Result<Vec<_>, _>>();
        let pv = pv.map_err(|error| {
            self.poisoned = true;
            TreeError::Backend(error)
        })?;
        Ok((packet.score, pv, stats))
    }
}
impl Drop for HipSelectiveTree {
    fn drop(&mut self) {
        // SAFETY: unique live context, destroyed before its library is unloaded.
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}
