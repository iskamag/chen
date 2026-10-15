//! Canonical JPEG engine entry point. Legacy minimax controls are archived separately.
fn main() {
    if let Err(error) = chen_jpeg2::run(std::env::args().collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
