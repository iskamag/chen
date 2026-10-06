//! Fetches the CC0 Lichess `chess-openings` dataset into `OUT_DIR` for the
//! runtime opening book. Only the SAN move lines are consumed; `src/opening.rs`
//! replays them through Chen's rules and validates them at load time.
//!
//! The revision is pinned, so the URLs are content-addressed by commit. The
//! fetch is best effort: on failure the volumes are left empty, the engine
//! still builds, and the book reports itself as unavailable. `curl`/`wget`
//! honor the standard `https_proxy`/`http_proxy` environment, so no proxy is
//! baked in here.
use std::{env, fs, path::Path, process::Command};

const REVISION: &str = "a6189a30dc273ccb21fc2536a9a2fefd5592a67a";
const DEFAULT_BASE: &str = "https://raw.githubusercontent.com/lichess-org/chess-openings";
const VOLUMES: [&str; 5] = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CHEN_OPENINGS_URL");
    let out = Path::new(&env::var("OUT_DIR").expect("OUT_DIR is set")).join("chen-openings");
    if let Err(error) = fs::create_dir_all(&out) {
        println!("cargo:warning=cannot create {}: {error}", out.display());
        return;
    }
    // A non-empty volume means a previous run fetched the whole set.
    if VOLUMES.iter().all(|name| is_fetched(&out.join(name))) {
        return;
    }
    let base =
        env::var("CHEN_OPENINGS_URL").unwrap_or_else(|_| format!("{DEFAULT_BASE}/{REVISION}"));
    // A failed attempt for this exact base is not retried, so an offline build
    // does not re-download on every invocation. Changing the URL retries.
    let marker = out.join(".attempted");
    if fs::read_to_string(&marker).ok().as_deref() != Some(base.as_str()) {
        match fetch(&base, &out) {
            Ok(()) => {
                let _ = fs::remove_file(&marker);
                println!("cargo:warning=opening book fetched into {}", out.display());
            }
            Err(error) => {
                let _ = fs::write(&marker, &base);
                println!(
                    "cargo:warning=opening book unavailable ({error}); Chen builds without one"
                );
            }
        }
    }
    // Always leave a file per volume so `include_str!` compiles; an empty
    // volume makes `OpeningBook::bundled` report the book as unavailable.
    for name in VOLUMES {
        let path = out.join(name);
        if !path.exists() {
            let _ = fs::write(&path, b"");
        }
    }
}

fn is_fetched(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.len() > 0)
}

fn fetch(base: &str, out: &Path) -> Result<(), String> {
    for name in VOLUMES {
        let url = format!("{base}/{name}");
        let bytes = download(&url)?;
        // Guard against caching an error page or an unrelated payload.
        if !bytes.starts_with(b"eco\t") {
            return Err(format!("{url} is not a chess-openings TSV"));
        }
        let path = out.join(name);
        fs::write(&path, bytes).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn download(url: &str) -> Result<Vec<u8>, String> {
    // curl and wget both read https_proxy/http_proxy from the environment.
    for (tool, args) in [
        ("curl", ["-fsSL", "--max-time", "60", url].as_slice()),
        ("wget", ["-q", "-O", "-", url].as_slice()),
    ] {
        if let Ok(output) = Command::new(tool).args(args).output()
            && output.status.success()
            && !output.stdout.is_empty()
        {
            return Ok(output.stdout);
        }
    }
    Err(format!("curl and wget could not download {url}"))
}
