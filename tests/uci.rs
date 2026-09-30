use chen::Position;
use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};
struct Engine {
    child: Child,
    input: ChildStdin,
    lines: Receiver<String>,
}
impl Engine {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_chen"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line.unwrap()).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            input,
            lines,
        }
    }
    fn send(&mut self, line: &str) {
        writeln!(self.input, "{line}").unwrap();
        self.input.flush().unwrap();
    }
    fn until(&self, prefix: &str) -> String {
        loop {
            let line = self
                .lines
                .recv_timeout(Duration::from_secs(10))
                .expect("UCI response timeout");
            if line.starts_with(prefix) {
                return line;
            }
        }
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
#[test]
fn uci_search_stop_options_and_recovery() {
    let mut engine = Engine::new();
    engine.send("uci");
    engine.until("uciok");
    engine.send("isready");
    engine.until("readyok");
    engine.send("setoption name Hash value 1");
    engine.send("setoption name Eval value material");
    engine.send("setoption name Quiescence value false");
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go depth 3");
    let best = engine.until("bestmove ");
    let mut p = Position::startpos();
    for text in ["e2e4", "e7e5"] {
        let mv = p.parse_uci_move(text).unwrap();
        p.make_move(mv);
    }
    assert!(
        p.parse_uci_move(best.split_whitespace().nth(1).unwrap())
            .is_ok()
    );
    engine.send("go infinite");
    engine.send("isready");
    engine.until("readyok");
    engine.send("stop");
    engine.until("bestmove ");
    // Repeated immediate stops exercise cancellation arriving before worker startup.
    for _ in 0..10 {
        engine.send("go infinite");
        engine.send("stop");
        engine.until("bestmove ");
    }
    engine.send("position startpos moves e2e5");
    engine.until("info string illegal move:");
    engine.send("d");
    assert_eq!(
        engine.until("info string "),
        format!("info string {}", p.to_fen())
    );
    engine.send("go nodes 25");
    assert!(
        engine
            .until("bestmove ")
            .split_whitespace()
            .nth(1)
            .is_some()
    );
    engine.send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    engine.send("go depth 2");
    assert_eq!(engine.until("bestmove "), "bestmove 0000");
    engine.send("quit");
}

#[test]
fn frontier_uci_and_resource_levels_are_selectable() {
    let mut engine = Engine::new();
    engine.send("uci");
    engine.until("uciok");
    engine.send("setoption name Backend value cpu-frontier");
    engine.send("setoption name BatchSize value 7");
    engine.send("setoption name Level value 1");
    engine.send("go depth 64");
    let info = engine.until("info depth ");
    assert!(info.starts_with("info depth 1 "));
    let best = engine.until("bestmove ");
    assert!(
        Position::startpos()
            .parse_uci_move(best.split_whitespace().nth(1).unwrap())
            .is_ok()
    );
    engine.send("setoption name Backend value bogus");
    engine.until("info string invalid or unsupported option");
    engine.send("go depth 1");
    assert!(
        engine
            .until("bestmove ")
            .split_whitespace()
            .nth(1)
            .is_some()
    );
    engine.send("quit");
}

#[test]
fn unavailable_requested_gpu_backend_never_substitutes_cpu() {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_chen"))
        .env("CHEN_GPU_LIBRARY", "/no/such/chen-test-library.so")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"setoption name Backend value hip-frontier\nsetoption name BatchSize value 8\ngo depth 1\n").unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(!result.status.success());
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("backend creation failed"));
    assert!(!stdout.contains("info depth"));
    assert!(!stdout.contains("bestmove"));
}
