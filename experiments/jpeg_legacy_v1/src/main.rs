//! Playable experiment. Hybrid delegates ordinary search explicitly; pure does not.
use chen::codec::CodecPosition;
use chen::codec_policy::Preferences;
use chen::{Move, Position, SearchConfig, SearchLimits, Searcher, Tapered};
use chen_relation_r1::{self as relation, Mode, Verdict};
use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy {
    Pure,
    Hybrid,
}
impl Policy {
    fn name(self) -> &'static str {
        if self == Self::Pure {
            "JPEG relation minimax"
        } else {
            "JPEG-assisted Chen PVS"
        }
    }
}
#[derive(Clone)]
struct Candidate {
    mv: Move,
    score: i32,
    terminal: Verdict,
    allows_mate: Option<bool>,
    draw: bool,
}
fn emit(s: impl AsRef<str>) {
    println!("{}", s.as_ref());
    io::stdout().flush().ok();
}
fn stopped(stop: &AtomicBool, deadline: Instant) -> bool {
    stop.load(Ordering::Relaxed) || Instant::now() >= deadline
}
fn draw(p: &Position, history: &[u64]) -> bool {
    p.halfmove_clock() >= 100
        || p.insufficient_material()
        || history
            .iter()
            .rev()
            .take(p.halfmove_clock() as usize)
            .filter(|&&h| h == p.hash())
            .count()
            >= 2
}
fn prepass(p: &Position, history: &[u64], stop: &AtomicBool, deadline: Instant) -> Vec<Candidate> {
    let codec = CodecPosition::from_position(p).relation();
    let preferences = Preferences::new(p, &codec);
    let mut candidates: Vec<_> = codec
        .decisions()
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            preferences.score(i).map(|s| Candidate {
                mv: d.mv,
                score: s.score,
                terminal: Verdict::Unknown,
                allows_mate: None,
                draw: false,
            })
        })
        .collect();
    candidates.sort_by_key(|c| std::cmp::Reverse(c.score));
    if candidates.is_empty() {
        return candidates;
    }
    let (base, actions) = relation::codec_root_actions(p);
    if stopped(stop, deadline) {
        return candidates;
    }
    let (labels, _) = relation::classify(&base, &actions, Mode::Stream, 4096);
    for c in &mut candidates {
        if let Some(l) = labels.iter().find(|x| x.mv == c.mv) {
            c.terminal = l.verdict;
        }
        let mut child = *p;
        child.make_move(c.mv);
        c.draw = draw(&child, history) || child.hash() == p.hash();
        if c.terminal == Verdict::Mate {
            c.score = 30_000;
        } else if c.terminal == Verdict::Stalemate || c.draw {
            c.score = 0;
        }
    }
    if candidates.iter().any(|c| c.terminal == Verdict::Mate) {
        return candidates;
    }
    for c in &mut candidates {
        if stopped(stop, deadline) {
            break;
        }
        if matches!(c.terminal, Verdict::Stalemate | Verdict::Mate) {
            c.allows_mate = Some(false);
            continue;
        }
        let action = actions.iter().find(|a| a.mv == c.mv).unwrap();
        let child = base.after(action);
        let replies = relation::legal_actions(&child);
        let checking = relation::checking_actions(&child, &replies);
        if stopped(stop, deadline) {
            break;
        }
        let (labels, _) = relation::classify(&child, &checking, Mode::Stream, 4096);
        if labels.iter().any(|l| l.verdict == Verdict::Mate) {
            c.allows_mate = Some(true);
        } else if labels.iter().all(|l| l.verdict != Verdict::Unknown) {
            c.allows_mate = Some(false);
        }
    }
    candidates.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.terminal == Verdict::Mate),
            match c.allows_mate {
                Some(false) => 0,
                None => 1,
                Some(true) => 2,
            },
            std::cmp::Reverse(c.score),
        )
    });
    candidates
}
struct RelationalMinimax<'a> {
    deadline: Instant,
    stop: &'a AtomicBool,
    nodes: u64,
}
impl RelationalMinimax<'_> {
    // Full-width minimax over exact codec alternatives. No alpha-beta and no beam.
    fn value(&mut self, p: &Position, history: &mut Vec<u64>, depth: u8, ply: i32) -> Option<i32> {
        if stopped(self.stop, self.deadline) {
            return None;
        }
        self.nodes += 1;
        let codec = CodecPosition::from_position(p).relation();
        let pref = Preferences::new(p, &codec);
        let mut ranked: Vec<_> = codec
            .decisions()
            .iter()
            .enumerate()
            .filter_map(|(i, d)| pref.score(i).map(|s| (d.mv, s.score)))
            .collect();
        if ranked.is_empty() {
            return Some(
                if codec
                    .attacked_before(p.side_to_move().opposite(), p.king_square(p.side_to_move()))
                {
                    -30_000 + ply
                } else {
                    0
                },
            );
        }
        if draw(p, history) {
            return Some(0);
        }
        ranked.sort_by_key(|x| std::cmp::Reverse(x.1));
        let (base, actions) = relation::root_actions(p);
        let (labels, _) = relation::classify(&base, &actions, Mode::Stream, 4096);
        if let Some(m) = labels.iter().find(|l| l.verdict == Verdict::Mate) {
            let _ = m;
            return Some(30_000 - ply - 1);
        }
        let mut best = -32_000;
        for (mv, static_score) in ranked {
            if stopped(self.stop, self.deadline) {
                return None;
            }
            let terminal = labels
                .iter()
                .find(|l| l.mv == mv)
                .map_or(Verdict::Unknown, |l| l.verdict);
            let mut child = *p;
            child.make_move(mv);
            history.push(p.hash());
            let score = if terminal == Verdict::Stalemate || draw(&child, history) {
                Some(0)
            } else if depth <= 1 {
                Some(static_score)
            } else {
                self.value(&child, history, depth - 1, ply + 1).map(|v| -v)
            };
            history.pop();
            best = best.max(score?);
        }
        Some(best)
    }
    fn iteration(
        &mut self,
        p: &Position,
        history: &[u64],
        candidates: &[Candidate],
        depth: u8,
    ) -> Option<(Move, i32)> {
        let known_safe = candidates.iter().any(|c| c.allows_mate == Some(false));
        let mut best = None;
        let mut path = history.to_vec();
        path.push(p.hash());
        for c in candidates {
            if known_safe && c.allows_mate != Some(false) {
                continue;
            }
            if stopped(self.stop, self.deadline) {
                return None;
            }
            let score = if c.terminal == Verdict::Mate {
                30_000
            } else if c.terminal == Verdict::Stalemate || c.draw {
                0
            } else if depth == 1 {
                c.score
            } else {
                let mut child = *p;
                child.make_move(c.mv);
                -self.value(&child, &mut path, depth - 1, 1)?
            };
            if best.is_none_or(|(_, v)| score > v) {
                best = Some((c.mv, score));
            }
        }
        best
    }
}
fn choose(
    p: Position,
    history: Vec<u64>,
    ms: u64,
    policy: Policy,
    searcher: Arc<Mutex<Searcher>>,
    stop: Arc<AtomicBool>,
) {
    let started = Instant::now();
    let total = Duration::from_millis(ms.saturating_sub(35).max(1));
    let deadline = started + total;
    let gate_deadline = deadline.min(started + Duration::from_millis(25));
    let candidates = prepass(&p, &history, &stop, gate_deadline);
    let prepass_ms = started.elapsed().as_millis();
    let Some(first) = candidates.first() else {
        emit("bestmove 0000");
        return;
    };
    let mut best = first.mv;
    let mut score = first.score;
    let mate = candidates.iter().find(|c| c.terminal == Verdict::Mate);
    if let Some(c) = mate {
        best = c.mv;
        score = 30_000;
        emit(format!(
            "info depth 1 score mate 1 time {} pv {}",
            started.elapsed().as_millis(),
            best
        ));
    } else if policy == Policy::Pure && !stopped(&stop, deadline) {
        let mut model = RelationalMinimax {
            deadline,
            stop: &stop,
            nodes: 0,
        };
        for depth in 1..=12 {
            let Some((mv, value)) = model.iteration(&p, &history, &candidates, depth) else {
                break;
            };
            best = mv;
            score = value;
            emit(format!(
                "info depth {depth} score cp {score} nodes {} time {} pv {}",
                model.nodes,
                started.elapsed().as_millis(),
                best
            ));
            if score.abs() >= 29_800 {
                break;
            }
        }
    } else if policy == Policy::Hybrid && !stopped(&stop, deadline) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let mut board = p;
        let mut engine = searcher.lock().unwrap();
        let result = engine.search(
            &mut board,
            SearchLimits {
                depth: 64,
                nodes: None,
                time: Some(remaining),
            },
            &history,
            |r| emit(r.uci_info()),
        );
        if let Some(mv) = result.best_move
            && let Some(c) = candidates.iter().find(|c| c.mv == mv)
            && (c.allows_mate == Some(false)
                || !candidates.iter().any(|a| a.allows_mate == Some(false)))
        {
            best = mv;
            score = result.score;
        }
    }
    emit(format!(
        "info string policy={} prepass_candidates={} known_safe={} known_mate_losses={} selected_score={} elapsed_ms={} prepass_ms={}",
        policy.name(),
        candidates.len(),
        candidates
            .iter()
            .filter(|c| c.allows_mate == Some(false))
            .count(),
        candidates
            .iter()
            .filter(|c| c.allows_mate == Some(true))
            .count(),
        score,
        started.elapsed().as_millis(),
        prepass_ms
    ));
    emit(format!("bestmove {best}"));
}
fn parse_position(line: &str) -> Result<(Position, Vec<u64>), String> {
    let words: Vec<_> = line.split_whitespace().collect();
    if words.len() < 2
        || (words.get(1) != Some(&"startpos") && (words.get(1) != Some(&"fen") || words.len() < 8))
    {
        return Err("expected startpos or six-field fen".into());
    }
    let mut index;
    if words.get(1) == Some(&"startpos") {
        index = 2;
    } else {
        index = 8;
    }
    let mut p = if words.get(1) == Some(&"startpos") {
        Position::startpos()
    } else {
        Position::from_fen(&words[2..index].join(" "))?
    };
    let mut history = Vec::new();
    if words.get(index) == Some(&"moves") {
        index += 1;
        for text in &words[index..] {
            let mv = p.parse_uci_move(text)?;
            history.push(p.hash());
            p.make_move(mv);
        }
    }
    Ok((p, history))
}
fn budget(line: &str, p: &Position) -> u64 {
    let w: Vec<_> = line.split_whitespace().collect();
    let get = |key: &str| {
        w.iter()
            .position(|&x| x == key)
            .and_then(|i| w.get(i + 1))
            .and_then(|v| v.parse::<u64>().ok())
    };
    if let Some(ms) = get("movetime") {
        return ms;
    }
    let white = p.side_to_move() == chen::Color::White;
    let remaining = get(if white { "wtime" } else { "btime" });
    if let Some(time) = remaining {
        let inc = get(if white { "winc" } else { "binc" }).unwrap_or(0);
        return (time / get("movestogo").unwrap_or(30).max(1) + inc * 3 / 4)
            .min(time.saturating_sub(50).max(1))
            .max(1);
    }
    1000
}
fn finish(job: &mut Option<thread::JoinHandle<()>>, stop: &AtomicBool) {
    if let Some(handle) = job.take() {
        stop.store(true, Ordering::Relaxed);
        handle.join().ok();
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut policy = if args.windows(2).any(|w| w == ["--mode", "pure"]) {
        Policy::Pure
    } else {
        Policy::Hybrid
    };
    let config = SearchConfig {
        pvs: true,
        aspiration_window: 25,
        hash_mb: 16,
        ..SearchConfig::default()
    };
    let engine = Searcher::new(Box::new(Tapered), config);
    let stop = engine.stop_handle();
    let searcher = Arc::new(Mutex::new(engine));
    let mut job = None;
    let mut position = Position::startpos();
    let mut history = Vec::new();
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let line = line.trim();
        if line == "uci" {
            emit(format!("id name {}", policy.name()));
            emit("id author Chen architecture experiments");
            emit(format!(
                "option name Policy type combo default {} var hybrid var pure",
                if policy == Policy::Pure {
                    "pure"
                } else {
                    "hybrid"
                }
            ));
            emit("option name Hash type spin default 16 min 1 max 1024");
            emit("option name Threads type spin default 1 min 1 max 1");
            emit("uciok");
        } else if line == "isready" {
            emit("readyok");
        } else if line == "stop" {
            finish(&mut job, &stop);
        } else if line == "quit" {
            finish(&mut job, &stop);
            break;
        } else if line == "ucinewgame" {
            finish(&mut job, &stop);
            searcher.lock().unwrap().clear();
            history.clear();
        } else if line.starts_with("position ") {
            finish(&mut job, &stop);
            match parse_position(line) {
                Ok((p, h)) => {
                    position = p;
                    history = h;
                }
                Err(e) => emit(format!("info string position error: {e}")),
            }
        } else if line.starts_with("setoption ") {
            finish(&mut job, &stop);
            if line.contains("name Policy value pure") {
                policy = Policy::Pure;
            } else if line.contains("name Policy value hybrid") {
                policy = Policy::Hybrid;
            } else if let Some(value) = line
                .strip_prefix("setoption name Hash value ")
                .and_then(|v| v.parse::<usize>().ok())
            {
                let mut e = searcher.lock().unwrap();
                let mut c = e.config().clone();
                c.hash_mb = value.clamp(1, 1024);
                e.set_config(c);
            }
        } else if line.starts_with("go") {
            finish(&mut job, &stop);
            stop.store(false, Ordering::Relaxed);
            let ms = budget(line, &position);
            let e = searcher.clone();
            let s = stop.clone();
            let h = history.clone();
            let p = position;
            job = Some(thread::spawn(move || choose(p, h, ms, policy, e, s)));
        }
    }
    finish(&mut job, &stop);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn inspect(fen: &str) -> (Position, Vec<Candidate>) {
        let p = Position::from_fen(fen).unwrap();
        let c = prepass(
            &p,
            &[],
            &AtomicBool::new(false),
            Instant::now() + Duration::from_secs(2),
        );
        (p, c)
    }
    #[test]
    fn mate_one_both_colors_even_at_draw_clock() {
        for fen in [
            "7k/5Q2/6K1/8/8/8/8/8 w - - 100 1",
            "8/8/8/8/8/6k1/5q2/7K b - - 100 1",
        ] {
            let (p, c) = inspect(fen);
            let mate = c.iter().find(|c| c.terminal == Verdict::Mate).unwrap();
            let mut child = p;
            child.make_move(mate.mv);
            assert!(!child.has_legal_move());
            assert!(child.is_in_check(child.side_to_move()));
            assert_eq!(mate.score, 30_000);
        }
    }
    #[test]
    fn opposing_mate_one_is_rejected_both_colors() {
        for (fen, losing) in [
            (
                "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2",
                "g2g4",
            ),
            (
                "rnbqkbnr/ppppp1pp/5p2/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
                "g7g5",
            ),
        ] {
            let (p, c) = inspect(fen);
            let bad = p.parse_uci_move(losing).unwrap();
            assert_eq!(
                c.iter().find(|c| c.mv == bad).unwrap().allows_mate,
                Some(true)
            );
            assert_eq!(c[0].allows_mate, Some(false));
        }
    }
    #[test]
    fn repetition_history_and_fifty_move_draw_are_preserved() {
        let (p, h) =
            parse_position("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8")
                .unwrap();
        assert!(draw(&p, &h));
        assert_eq!(h.len(), 8);
        let p = Position::from_fen("7k/8/8/8/8/8/8/R6K w - - 100 1").unwrap();
        assert!(draw(&p, &[]));
        let (p, c) = inspect("7k/8/8/8/8/8/8/R6K w - - 99 1");
        assert!(
            c.iter()
                .filter(|c| c.terminal != Verdict::Mate)
                .all(|c| c.draw)
        );
        assert!(!draw(&p, &[]));
        assert!(parse_position("position fen bad").is_err());
    }
    #[test]
    fn exhausted_deadline_keeps_legal_fallback_without_claiming_certificate() {
        let p = Position::startpos();
        let c = prepass(&p, &[], &AtomicBool::new(false), Instant::now());
        assert_eq!(c.len(), 20);
        assert!(
            c.iter()
                .all(|c| c.terminal == Verdict::Unknown && c.allows_mate.is_none())
        );
        let stop = AtomicBool::new(false);
        let mut model = RelationalMinimax {
            deadline: Instant::now(),
            stop: &stop,
            nodes: 0,
        };
        assert!(model.iteration(&p, &[], &c, 4).is_none());
    }
}
