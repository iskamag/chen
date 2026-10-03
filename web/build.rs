use std::process::Command;
fn main() {
    let mut revision = Command::new("git")
        .current_dir("..")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    if Command::new("git")
        .current_dir("..")
        .args([
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
        ])
        .status()
        .is_ok_and(|status| !status.success())
    {
        revision = format!("{}+dirty", revision.trim());
    }
    println!("cargo:rustc-env=CHEN_REVISION={}", revision.trim());
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/heads/master");
}
