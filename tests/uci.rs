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
    /// Collect every line up to and including the first starting with `prefix`.
    fn collect_until(&self, prefix: &str) -> Vec<String> {
        let mut lines = Vec::new();
        loop {
            let line = self
                .lines
                .recv_timeout(Duration::from_secs(10))
                .expect("UCI response timeout");
            let done = line.starts_with(prefix);
            lines.push(line);
            if done {
                return lines;
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
    engine.send("setoption name PVS value true");
    engine.send("setoption name AspirationWindow value 1");
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
    engine.send("setoption name Backend value cpu-resident");
    engine.send("go depth 64");
    assert!(engine.until("info depth ").starts_with("info depth 1 "));
    assert!(
        Position::startpos()
            .parse_uci_move(engine.until("bestmove ").split_whitespace().nth(1).unwrap())
            .is_ok()
    );
    engine.send("quit");
}

#[test]
fn unavailable_requested_gpu_backend_never_substitutes_cpu() {
    use std::io::Write;
    for backend in ["hip-frontier", "hip-resident", "hip-selective"] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_chen"))
            .env("CHEN_GPU_LIBRARY", "/no/such/chen-test-library.so")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(format!("setoption name Backend value {backend}\nsetoption name BatchSize value 8\ngo depth 1\n").as_bytes()).unwrap();
        let result = child.wait_with_output().unwrap();
        assert!(!result.status.success());
        let stdout = String::from_utf8(result.stdout).unwrap();
        assert!(stdout.contains("backend creation failed"));
        assert!(!stdout.contains("info depth"));
        assert!(!stdout.contains("bestmove"));
    }
}

#[test]
fn selective_uci_reports_mixed_depth_and_recovers_after_invalid_option() {
    let mut engine = Engine::new();
    engine.send("setoption name Backend value cpu-selective");
    engine.send("setoption name SelectiveBatch value 1");
    engine.send("setoption name SelectiveCandidates value 1");
    engine.send("setoption name SelectiveBatch value 2");
    engine.until("info string invalid or unsupported option");
    engine.send("go depth 3 nodes 41");
    let first = engine.until("info depth ");
    assert!(first.starts_with("info depth 1 seldepth 1"));
    let second = engine.until("info depth ");
    assert!(second.starts_with("info depth 1 seldepth 2"));
    let best = engine.until("bestmove ");
    assert!(
        Position::startpos()
            .parse_uci_move(best.split_whitespace().nth(1).unwrap())
            .is_ok()
    );
    engine.send("setoption name SelectiveCandidates value 1024");
    engine.send("setoption name SelectiveBatch value 8");
    engine.send("go depth 1 nodes 21");
    assert!(
        engine
            .until("info depth ")
            .starts_with("info depth 1 seldepth 1")
    );
    engine.until("bestmove ");
    engine.send("quit");
}

#[test]
fn opening_book_selects_legal_moves_and_toggles() {
    let mut engine = Engine::new();
    engine.send("uci");
    engine.until("uciok");
    engine.send("setoption name Book value true");
    engine.send("position startpos");
    engine.send("go depth 8");
    let lines = engine.collect_until("bestmove ");
    let info = lines
        .iter()
        .find(|l| l.contains(" book "))
        .expect("book info line");
    assert!(
        info.contains("depth 0") && info.contains("score cp 0"),
        "{info}"
    );
    let move_text = lines
        .last()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_string();
    let start = Position::startpos();
    assert!(start.parse_uci_move(&move_text).is_ok(), "{move_text}");

    // The book also answers deeper in the tree, following the played history.
    let mut position = Position::startpos();
    let first = position.parse_uci_move(&move_text).unwrap();
    position.make_move(first);
    // Chen is the side to move; let the opponent reply, then Chen's next move.
    let reply = position.legal_moves()[0];
    position.make_move(reply);
    engine.send(&format!("position startpos moves {} {}", first, reply));
    engine.send("go depth 8");
    let lines = engine.collect_until("bestmove ");
    let next = lines.last().unwrap().split_whitespace().nth(1).unwrap();
    assert!(position.parse_uci_move(next).is_ok(), "{next}");

    // Disabling the book restores a normal searched move.
    engine.send("setoption name Book value false");
    engine.send("position startpos");
    engine.send("go depth 3");
    let lines = engine.collect_until("bestmove ");
    assert!(lines.iter().all(|l| !l.contains(" book ")), "{lines:?}");
    assert!(
        lines.iter().any(|l| l.starts_with("info depth 3 ")),
        "{lines:?}"
    );
    engine.send("quit");
}

#[test]
fn drowsy_uci_controls_telemetry_and_infinite_attention() {
    let mut engine = Engine::new();
    engine.send("setoption name Backend value drowsy");
    engine.send("setoption name DrowsyDelta value 250");
    engine.send("setoption name DrowsyTheta value 0.55");
    engine.send("position startpos moves e2e4");
    engine.send("go nodes 100 depth 4");
    let stats = engine.until("info string drowsy ");
    assert!(stats.contains("fail-meow ") && stats.contains("fail-zoomies "));
    let best = engine.until("bestmove ");
    let mut position = Position::startpos();
    position.make_move(position.parse_uci_move("e2e4").unwrap());
    assert!(
        position
            .parse_uci_move(best.split_whitespace().nth(1).unwrap())
            .is_ok()
    );
    engine.send("setoption name DrowsyTheta value inf");
    engine.send("go nodes 100 depth 4");
    let stats = engine.until("info string drowsy ");
    assert!(stats.contains("walkers 0"));
    assert_eq!(
        engine.until("bestmove "),
        format!("bestmove {}", position.legal_moves()[0])
    );
    engine.send("setoption name DrowsyTheta value NaN");
    engine.until("info string invalid or unsupported option");
    engine.send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    engine.send("go depth 2");
    engine.until("info string drowsy ");
    assert_eq!(engine.until("bestmove "), "bestmove 0000");
    engine.send("quit");
}
