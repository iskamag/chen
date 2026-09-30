//! Linux runtime loading of Chen's versioned HIP ABI; no ROCm build dependency in Rust.
use crate::{
    Color, Position,
    backend::{BatchEvaluator, BatchTelemetry},
    core::PIECES,
};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::Path,
    ptr::NonNull,
};
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}
#[repr(C)]
struct PackedPosition {
    pieces: [u64; 12],
    side: u32,
    reserved: u32,
}
impl From<&Position> for PackedPosition {
    fn from(p: &Position) -> Self {
        let mut pieces = [0; 12];
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                pieces[color.index() * 6 + piece.index()] = p.pieces(color, piece);
            }
        }
        Self {
            pieces,
            side: p.side_to_move().index() as u32,
            reserved: 0,
        }
    }
}
#[repr(C)]
struct DeviceInfo {
    name: [c_char; 128],
    architecture: [c_char; 64],
    total_memory: u64,
    compute_units: u32,
    wave_size: u32,
    async_engines: u32,
    runtime_version: u32,
}
#[repr(C)]
#[derive(Default)]
struct Timings {
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
    count: u64,
    capacity: u64,
}
type Create = unsafe extern "C" fn(i32, u64, *mut c_char, u64) -> *mut c_void;
type GetInfo = unsafe extern "C" fn(*mut c_void, *mut DeviceInfo, *mut c_char, u64) -> i32;
type Evaluate = unsafe extern "C" fn(
    *mut c_void,
    *const PackedPosition,
    *mut i32,
    u64,
    u32,
    *mut Timings,
    *mut c_char,
    u64,
) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
pub(crate) struct Library(NonNull<c_void>);
impl Library {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        use std::os::unix::ffi::OsStrExt;
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "HIP library path contains NUL")?;
        // SAFETY: filename is NUL-terminated; ownership ends at Library::drop.
        let ptr = unsafe { dlopen(path.as_ptr(), 2) };
        NonNull::new(ptr)
            .map(Self)
            .ok_or_else(|| format!("cannot load HIP library: {}", loader_error()))
    }
    pub(crate) fn symbol(&self, name: &CStr) -> Result<*mut c_void, String> {
        // SAFETY: the library is live and the requested symbol is NUL-terminated.
        let ptr = unsafe { dlsym(self.0.as_ptr(), name.as_ptr()) };
        if ptr.is_null() {
            Err(format!("missing HIP symbol: {}", name.to_string_lossy()))
        } else {
            Ok(ptr)
        }
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: unique ownership of a successful dlopen handle.
        unsafe {
            dlclose(self.0.as_ptr());
        }
    }
}
fn loader_error() -> String {
    // SAFETY: dlerror returns either null or its own NUL-terminated error string.
    let ptr = unsafe { dlerror() };
    if ptr.is_null() {
        "unknown dynamic loader error".into()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}
pub(crate) fn buffer_string(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
#[derive(Clone, Debug)]
pub struct DeviceDescription {
    pub name: String,
    pub architecture: String,
    pub total_memory: u64,
    pub hip_multiprocessors: u32,
    pub wave_size: u32,
    pub async_engines: u32,
    pub runtime_version: u32,
}
pub struct HipBatchEvaluator {
    _library: Library,
    context: NonNull<c_void>,
    evaluate: Evaluate,
    destroy: Destroy,
    capacity: usize,
    evaluator: u32,
    name: String,
    device: DeviceDescription,
}
// SAFETY: exclusive context ownership; evaluate requires &mut self. The ABI restores
// the HIP device on calls and permits serialized transfer between host threads.
unsafe impl Send for HipBatchEvaluator {}
impl HipBatchEvaluator {
    pub fn new(path: impl AsRef<Path>, evaluator: &str, capacity: usize) -> Result<Self, String> {
        let evaluator_id = match evaluator {
            "material" => 0,
            "tapered" => 1,
            _ => return Err("HIP baseline supports material or tapered".into()),
        };
        let library = Library::open(path.as_ref())?;
        // SAFETY: these signatures are the versioned declarations in gpu/chen_gpu.h.
        let (abi, create, info, evaluate, destroy) = unsafe {
            (
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> u32>(
                    library.symbol(c"chen_gpu_abi_version")?,
                ),
                std::mem::transmute::<*mut c_void, Create>(library.symbol(c"chen_gpu_create")?),
                std::mem::transmute::<*mut c_void, GetInfo>(
                    library.symbol(c"chen_gpu_device_info")?,
                ),
                std::mem::transmute::<*mut c_void, Evaluate>(library.symbol(c"chen_gpu_evaluate")?),
                std::mem::transmute::<*mut c_void, Destroy>(library.symbol(c"chen_gpu_destroy")?),
            )
        };
        // SAFETY: ABI version function takes no data; other functions are not called before this check.
        if unsafe { abi() } != 1 {
            return Err("unsupported Chen HIP ABI (expected 1)".into());
        }
        let mut error = [0 as c_char; 512];
        // SAFETY: buffer is writable for its advertised size; capacity is checked by C ABI.
        let context = NonNull::new(unsafe {
            create(0, capacity as u64, error.as_mut_ptr(), error.len() as u64)
        })
        .ok_or_else(|| buffer_string(&error))?;
        let mut raw = DeviceInfo {
            name: [0; 128],
            architecture: [0; 64],
            total_memory: 0,
            compute_units: 0,
            wave_size: 0,
            async_engines: 0,
            runtime_version: 0,
        };
        // SAFETY: valid context, correctly sized writable ABI data and error buffer.
        if unsafe {
            info(
                context.as_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        } != 0
        {
            // SAFETY: unique freshly created context, library still loaded.
            unsafe {
                destroy(context.as_ptr());
            }
            return Err(buffer_string(&error));
        }
        let device = DeviceDescription {
            name: buffer_string(&raw.name),
            architecture: buffer_string(&raw.architecture),
            total_memory: raw.total_memory,
            hip_multiprocessors: raw.compute_units,
            wave_size: raw.wave_size,
            async_engines: raw.async_engines,
            runtime_version: raw.runtime_version,
        };
        Ok(Self {
            _library: library,
            context,
            evaluate,
            destroy,
            capacity,
            evaluator: evaluator_id,
            name: format!("hip-{evaluator}"),
            device,
        })
    }
    pub fn from_environment(evaluator: &str, capacity: usize) -> Result<Self, String> {
        Self::new(
            std::env::var_os("CHEN_GPU_LIBRARY")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| "gpu/build/libchen_gpu.so".into()),
            evaluator,
            capacity,
        )
    }
    pub fn device(&self) -> &DeviceDescription {
        &self.device
    }
}
impl BatchEvaluator for HipBatchEvaluator {
    fn name(&self) -> &str {
        &self.name
    }
    fn evaluate_batch(
        &mut self,
        positions: &[Position],
        scores: &mut [i32],
    ) -> Result<BatchTelemetry, String> {
        if positions.len() != scores.len() {
            return Err("batch input/output length mismatch".into());
        }
        if positions.len() > self.capacity {
            return Err("batch exceeds HIP context capacity".into());
        }
        let packed: Vec<_> = positions.iter().map(PackedPosition::from).collect();
        let mut timing = Timings::default();
        let mut error = [0 as c_char; 512];
        // SAFETY: context is exclusively borrowed and live; buffers have identical count;
        // the synchronous ABI finishes all access before returning. No aliases to scores.
        let code = unsafe {
            (self.evaluate)(
                self.context.as_ptr(),
                packed.as_ptr(),
                scores.as_mut_ptr(),
                packed.len() as u64,
                self.evaluator,
                &mut timing,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        };
        if code != 0 {
            return Err(format!("HIP evaluation failed: {}", buffer_string(&error)));
        }
        Ok(BatchTelemetry {
            positions: timing.count,
            h2d_ms: timing.h2d_ms,
            kernel_ms: timing.kernel_ms,
            d2h_ms: timing.d2h_ms,
        })
    }
}
impl Drop for HipBatchEvaluator {
    fn drop(&mut self) {
        // SAFETY: unique context; library field drops only after this method.
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_layout_matches_header() {
        assert_eq!(std::mem::size_of::<PackedPosition>(), 104);
        assert_eq!(std::mem::size_of::<DeviceInfo>(), 216);
        assert_eq!(std::mem::size_of::<Timings>(), 40);
    }
    #[test]
    fn missing_library_is_an_error() {
        assert!(HipBatchEvaluator::new("/no/such/chen.so", "material", 1).is_err());
    }
}
