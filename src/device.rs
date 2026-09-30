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
        if capacity == 0 {
            return Err("GPU tree capacity must be positive".into());
        }
        let library = Library::open(path.as_ref())?;
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
