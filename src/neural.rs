//! Version-pinned same-model inference plugin, composed with the resident arena.
//! The portable engine retains no Torch dependency; HIP builds load it on request.
use crate::{
    Move,
    device::HipRules,
    hip::{Library, buffer_string},
    resident::{ResidentTree, TreeError, TreeSpan, TreeStats},
    rules::RuleState,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, c_char, c_void},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    ptr::NonNull,
};
const ARCHIVE_LIMIT: u64 = 64 * 1024 * 1024;
const MANIFEST_LIMIT: u64 = 65536;

fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.is_empty() || bytes.len() as u64 > limit {
        return Err(format!("{}: empty or oversized artifact", path.display()));
    }
    Ok(bytes)
}
fn appended(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    name.into()
}
fn configured_libraries() -> Result<(PathBuf, PathBuf), String> {
    let neural = std::env::var_os("CHEN_NEURAL_LIBRARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| "gpu/build/libchen_neural.so".into());
    let runtime = match std::env::var_os("CHEN_NEURAL_HIP_RUNTIME") {
        Some(path) => PathBuf::from(path),
        None => {
            let bytes = bounded_read(&appended(&neural, ".runtime"), 8192)?;
            let path = std::str::from_utf8(&bytes)
                .map_err(|e| e.to_string())?
                .trim();
            if path.is_empty() {
                return Err("empty neural HIP runtime path".into());
            }
            PathBuf::from(path)
        }
    };
    Ok((neural, runtime))
}

/// Keep the producing runtime loaded for the whole CLI session, before any
/// classical HIP context can bind another runtime through its SONAME.
pub struct ConfiguredRuntime {
    _library: Library,
}
pub fn preload_configured_runtime() -> Result<Option<ConfiguredRuntime>, String> {
    if std::env::var_os("CHEN_NEURAL_MODEL").is_none() {
        return Ok(None);
    }
    let (_, runtime) = configured_libraries()?;
    Ok(Some(ConfiguredRuntime {
        _library: Library::open_global(&runtime)?,
    }))
}
fn exact(manifest: &Value, key: &str, expected: Value) -> Result<(), String> {
    if manifest.get(key) != Some(&expected) {
        return Err(format!("unsupported inference manifest field: {key}"));
    }
    Ok(())
}

/// Read once, verify hashes and the exact embedded-manifest bytes, then keep
/// the archive bytes owned until native deserialization completes. No path race.
pub struct ModelArtifact {
    bytes: Vec<u8>,
    manifest_json: String,
    pub sha256: String,
    pub max_batch: u64,
    pub manifest: Value,
}
impl ModelArtifact {
    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = bounded_read(path, ARCHIVE_LIMIT)?;
        let sidecar = bounded_read(&appended(path, ".json"), MANIFEST_LIMIT)?;
        let envelope: Value = serde_json::from_slice(&sidecar).map_err(|e| e.to_string())?;
        let expected = envelope
            .get("archive_sha256")
            .and_then(Value::as_str)
            .ok_or("missing archive_sha256")?;
        let sha256 = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if expected != sha256 {
            return Err("inference archive SHA256 mismatch".into());
        }
        let manifest_json = envelope
            .get("manifest_json")
            .and_then(Value::as_str)
            .ok_or("missing exact manifest_json")?
            .to_owned();
        if manifest_json.is_empty() || manifest_json.len() as u64 > MANIFEST_LIMIT {
            return Err("invalid bounded embedded manifest".into());
        }
        let manifest: Value = serde_json::from_str(&manifest_json).map_err(|e| e.to_string())?;
        if envelope.get("manifest") != Some(&manifest) || !manifest.is_object() {
            return Err("sidecar and embedded manifest disagree".into());
        }
        for (key, value) in [
            ("deployment_version", serde_json::json!(1)),
            ("feature_schema", serde_json::json!(1)),
            ("input_dtype", serde_json::json!("float32")),
            ("input_shape", serde_json::json!(["B", 18, 8, 8])),
            ("output_dtype", serde_json::json!("int32")),
            ("output_shape", serde_json::json!(["B"])),
            (
                "target_kind",
                serde_json::json!("finite_depth_minimax_cp_utility"),
            ),
            ("epsilon", serde_json::json!(1e-6)),
            ("perspective", serde_json::json!("side_to_move")),
            ("rounding", serde_json::json!("nearest_even")),
            ("terminal_responsibility", serde_json::json!("simulator")),
        ] {
            exact(&manifest, key, value)?;
        }
        if manifest.get("scale_cp").and_then(Value::as_f64) != Some(800.0) {
            return Err("unsupported inference utility scale".into());
        }
        exact(
            &manifest,
            "format",
            serde_json::json!("torchscript_static_reference"),
        )?;
        exact(&manifest, "score_limit", serde_json::json!(29000))?;
        exact(
            &manifest,
            "head_readout",
            serde_json::json!("softmax_logits_2_minus_0"),
        )?;
        exact(&manifest, "strength_evidence", serde_json::json!(false))?;
        if !manifest.get("provenance").is_some_and(Value::is_object) {
            return Err("missing deployment provenance".into());
        }
        for key in ["state_sha256", "checkpoint_sha256"] {
            if !manifest.get(key).and_then(Value::as_str).is_some_and(|s| {
                s.len() == 64
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }) {
                return Err(format!("invalid deployment fingerprint: {key}"));
            }
        }
        if !manifest
            .get("torch_version")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
        {
            return Err("missing export Torch version".into());
        }
        let max_batch = manifest
            .get("max_batch")
            .and_then(Value::as_u64)
            .filter(|n| (1..=4096).contains(n))
            .ok_or("invalid neural max_batch")?;
        let config = manifest
            .get("model_config")
            .and_then(Value::as_object)
            .ok_or("missing model configuration")?;
        if config.len() != 4
            || !matches!(
                config.get("name").and_then(Value::as_str),
                Some("square" | "relation" | "plan")
            )
        {
            return Err("invalid deployed model configuration".into());
        }
        for (key, low, high) in [("width", 8, 512), ("depth", 1, 32), ("plan_tokens", 1, 128)] {
            if !config
                .get(key)
                .and_then(Value::as_u64)
                .is_some_and(|n| (low..=high).contains(&n))
            {
                return Err(format!("invalid deployed {key}"));
            }
        }
        Ok(Self {
            bytes,
            manifest_json,
            sha256,
            max_batch,
            manifest,
        })
    }
}
type Evaluate = unsafe extern "C" fn(
    *mut c_void,
    i32,
    *const RuleState,
    u64,
    *mut i32,
    *mut c_char,
    u64,
) -> i32;
#[repr(C)]
pub(crate) struct ExternalEvaluator {
    abi_version: u32,
    struct_size: u32,
    owner: *mut c_void,
    evaluate: Evaluate,
    runtime_identity: [u64; 5],
    max_batch: u64,
    reserved: u64,
}
type Create = unsafe extern "C" fn(
    *const c_void,
    u64,
    *const c_char,
    u64,
    i32,
    u64,
    *mut c_char,
    u64,
) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
struct NeuralEvaluator {
    context: NonNull<c_void>,
    destroy: Destroy,
    evaluate: Evaluate,
    identity: [u64; 5],
    max_batch: u64,
    _library: Library,
    _runtime: Library,
}
// SAFETY: unique owner, all uses through &mut HipNeuralTree, synchronous native
// calls restore device; plugin context cannot be shared across host threads.
unsafe impl Send for NeuralEvaluator {}
impl NeuralEvaluator {
    fn new(path: &Path, runtime_path: &Path, artifact: &ModelArtifact) -> Result<Self, String> {
        // Load the verified producing runtime first, then the plugin, then rules.
        let runtime = Library::open_global(runtime_path)?;
        let library = Library::open_global(path)?;
        macro_rules! symbol {
            ($name:literal,$ty:ty) => {{
                // SAFETY: exact signatures in chen_neural.h ABI1.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(library.symbol($name)?) }
            }};
        }
        let abi = symbol!(c"chen_neural_abi_version", unsafe extern "C" fn() -> u32);
        // SAFETY: no-argument metadata query before context creation.
        if unsafe { abi() } != 1 {
            return Err("unsupported neural plugin ABI".into());
        }
        let version = symbol!(
            c"chen_neural_torch_version",
            unsafe extern "C" fn() -> *const c_char
        );
        // SAFETY: ABI returns a static NUL-terminated version string.
        let version = unsafe { version() };
        if version.is_null() {
            return Err("missing native Torch version".into());
        }
        let actual = unsafe { CStr::from_ptr(version) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let exported = artifact.manifest["torch_version"].as_str().unwrap();
        if exported.split('+').next() != Some(actual) {
            return Err(format!(
                "Torch export/native version mismatch: {exported} vs {actual}"
            ));
        }
        let identity_fn = symbol!(
            c"chen_neural_runtime_identity",
            unsafe extern "C" fn(u32) -> u64
        );
        let mut identity = [0; 5];
        for (i, name) in [
            c"hipPointerGetAttributes",
            c"hipMemGetAddressRange",
            c"hipMalloc",
            c"hipFree",
            c"hipStreamSynchronize",
        ]
        .into_iter()
        .enumerate()
        {
            identity[i] = unsafe { identity_fn(i as u32) };
            if identity[i] == 0 || identity[i] != runtime.symbol(name)? as u64 {
                return Err(format!(
                    "plugin does not bind producing HIP runtime at selector {i}"
                ));
            }
        }
        let create = symbol!(c"chen_neural_create", Create);
        let destroy = symbol!(c"chen_neural_destroy", Destroy);
        let evaluate = symbol!(c"chen_neural_evaluate", Evaluate);
        let mut error = [0 as c_char; 1024];
        // SAFETY: frozen owned archive/manifest bytes live throughout synchronous
        // deserialization; validated lengths and error buffer. Returned ctx owned.
        let context = NonNull::new(unsafe {
            create(
                artifact.bytes.as_ptr().cast(),
                artifact.bytes.len() as u64,
                artifact.manifest_json.as_ptr().cast(),
                artifact.manifest_json.len() as u64,
                0,
                artifact.max_batch,
                error.as_mut_ptr(),
                error.len() as u64,
            )
        })
        .ok_or_else(|| buffer_string(&error))?;
        Ok(Self {
            context,
            destroy,
            evaluate,
            identity,
            max_batch: artifact.max_batch,
            _library: library,
            _runtime: runtime,
        })
    }
    fn descriptor(&self) -> ExternalEvaluator {
        ExternalEvaluator {
            abi_version: 1,
            struct_size: 80,
            owner: self.context.as_ptr(),
            evaluate: self.evaluate,
            runtime_identity: self.identity,
            max_batch: self.max_batch,
            reserved: 0,
        }
    }
}
impl Drop for NeuralEvaluator {
    fn drop(&mut self) {
        // SAFETY: unique live context, both libraries remain loaded until after Drop.
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}
/// Owns a resident rules arena and a same-model inference context. Drop the arena
/// before unloading neural dependencies; only device pointers cross the callback.
pub struct HipNeuralTree {
    tree: HipRules,
    neural: NeuralEvaluator,
    pub model_sha256: String,
}
impl HipNeuralTree {
    pub fn new(
        rules_library: &Path,
        neural_library: &Path,
        runtime_library: &Path,
        model: &Path,
        capacity: usize,
    ) -> Result<Self, String> {
        let artifact = ModelArtifact::read(model)?;
        let neural = NeuralEvaluator::new(neural_library, runtime_library, &artifact)?;
        let tree = HipRules::new_with_runtime(rules_library, capacity, neural.identity)?;
        Ok(Self {
            tree,
            neural,
            model_sha256: artifact.sha256,
        })
    }
    pub fn from_environment(capacity: usize) -> Result<Self, String> {
        let (neural_library, runtime_library) = configured_libraries()?;
        let model = std::env::var_os("CHEN_NEURAL_MODEL")
            .map(PathBuf::from)
            .ok_or("Eval neural requires CHEN_NEURAL_MODEL")?;
        let rules_library = std::env::var_os("CHEN_GPU_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| "gpu/build/libchen_gpu.so".into());
        Self::new(
            &rules_library,
            &neural_library,
            &runtime_library,
            &model,
            capacity,
        )
    }
}
impl ResidentTree for HipNeuralTree {
    fn name(&self) -> &str {
        "hip-resident"
    }
    fn evaluator_artifact_sha256(&self) -> Option<&str> {
        Some(&self.model_sha256)
    }
    fn reset(&mut self, root: RuleState, history: &[u64]) -> Result<(), TreeError> {
        self.tree.reset(root, history)
    }
    fn classify(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError> {
        self.tree.classify(span, adjudicate)
    }
    fn expand(&mut self, span: TreeSpan, adjudicate: bool) -> Result<TreeStats, TreeError> {
        self.tree.expand(span, adjudicate)
    }
    fn evaluate(&mut self, span: TreeSpan, evaluator: u32) -> Result<TreeStats, TreeError> {
        if evaluator != 2 {
            return Err(TreeError::Backend(
                "neural tree requires external evaluator2".into(),
            ));
        }
        self.tree.evaluate_external(span, &self.neural.descriptor())
    }
    fn reduce(&mut self, span: TreeSpan) -> Result<TreeStats, TreeError> {
        self.tree.reduce(span)
    }
    fn finish(&mut self) -> Result<(i32, Vec<Move>, TreeStats), TreeError> {
        self.tree.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_descriptor_layout() {
        assert_eq!(std::mem::size_of::<ExternalEvaluator>(), 80);
        assert_eq!(
            std::mem::offset_of!(ExternalEvaluator, runtime_identity),
            24
        );
        assert_eq!(std::mem::offset_of!(ExternalEvaluator, max_batch), 64);
    }
    #[test]
    fn artifact_admission_rejects_tampering_and_wrong_contracts() {
        use serde_json::json;
        let root = std::env::temp_dir().join(format!(
            "chen-model-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        struct Remove(PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Remove(root.clone());
        let path = root.join("model.pt");
        let bytes = b"admission fixture, never deserialized as Torch";
        std::fs::write(&path, bytes).unwrap();
        let sha = Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let manifest = json!({
            "deployment_version":1,"feature_schema":1,"format":"torchscript_static_reference",
            "input_dtype":"float32","input_shape":["B",18,8,8],"output_dtype":"int32","output_shape":["B"],
            "target_kind":"finite_depth_minimax_cp_utility","scale_cp":800.0,"epsilon":1e-6,
            "perspective":"side_to_move","rounding":"nearest_even","terminal_responsibility":"simulator",
            "head_readout":"softmax_logits_2_minus_0","score_limit":29000,"max_batch":64,
            "model_config":{"name":"relation","width":16,"depth":1,"plan_tokens":8},
            "state_sha256":"0".repeat(64),"checkpoint_sha256":"1".repeat(64),
            "torch_version":"2.10.0+rocm7.0","strength_evidence":false,"provenance":{}});
        let write = |m: &Value, hash: &str| {
            let envelope =
                json!({"manifest":m,"manifest_json":m.to_string(),"archive_sha256":hash});
            std::fs::write(appended(&path, ".json"), envelope.to_string()).unwrap();
        };
        write(&manifest, &sha);
        let admitted = ModelArtifact::read(&path).unwrap();
        assert_eq!(admitted.max_batch, 64);
        assert_eq!(admitted.bytes, bytes);
        for (key, value) in [
            ("feature_schema", json!(2)),
            ("perspective", json!("white")),
            ("target_kind", json!("untrained_architecture_fixture")),
            ("scale_cp", json!(400)),
            ("rounding", json!("away_from_zero")),
            ("epsilon", json!(0)),
            ("strength_evidence", json!(true)),
            ("checkpoint_sha256", Value::Null),
            ("state_sha256", json!("f")),
            ("provenance", Value::Null),
            ("max_batch", json!(0)),
            ("max_batch", json!(4097)),
            ("input_shape", json!(["B", 17, 8, 8])),
            ("output_dtype", json!("float32")),
        ] {
            let mut wrong = manifest.clone();
            wrong[key] = value;
            write(&wrong, &sha);
            assert!(
                ModelArtifact::read(&path).is_err(),
                "accepted changed {key}"
            );
        }
        write(&manifest, &"0".repeat(64));
        assert!(ModelArtifact::read(&path).err().unwrap().contains("SHA256"));
        write(&manifest, &sha);
        std::fs::write(&path, b"changed archive").unwrap();
        assert!(ModelArtifact::read(&path).is_err());
        std::fs::write(&path, bytes).unwrap();
        let envelope = json!({"manifest":manifest,"manifest_json":"{}","archive_sha256":sha});
        std::fs::write(appended(&path, ".json"), envelope.to_string()).unwrap();
        assert!(ModelArtifact::read(&path).is_err());
    }
}
