fn main() {
    if let Err(error) = chen_jpeg2::run(std::env::args().collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
