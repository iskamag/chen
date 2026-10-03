use std::path::Path;
use std::process::Command;

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let repo = Path::new(&manifest).parent().unwrap();
    let mut revision =
        git(repo, &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    if Command::new("git")
        .current_dir(repo)
        .args([
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
        ])
        .output()
        .is_ok_and(|output| revision != "unknown" && !output.status.success())
    {
        revision = format!("{}+dirty", revision.trim());
    }
    println!("cargo:rustc-env=CHEN_REVISION={}", revision.trim());
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    let mut git_paths = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(branch) = git(repo, &["symbolic-ref", "-q", "HEAD"]) {
        git_paths.push(branch);
    }
    for git_path in git_paths {
        if let Some(path) = git(repo, &["rev-parse", "--git-path", &git_path]) {
            println!("cargo:rerun-if-changed={}", repo.join(path).display());
        }
    }
}
