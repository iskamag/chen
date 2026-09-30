use chen::{Position, SearchConfig, SearchLimits, Searcher, core::perft, eval::evaluator};
use std::{
    env,
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("chen: {error}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("uci") => uci(),
        Some("perft") | Some("divide") => {
            let depth = args
                .get(1)
                .ok_or("usage: chen perft|divide DEPTH [FEN]")?
                .parse::<u32>()
                .map_err(|_| "invalid depth")?;
            if depth > 10 {
                return Err("perft depth must be <= 10".into());
            }
            let mut p = if args.len() > 2 {
                Position::from_fen(&args[2..].join(" "))?
            } else {
                Position::startpos()
            };
            let start = Instant::now();
            let nodes = if args[0] == "divide" && depth > 0 {
                let mut total = 0;
                for mv in p.legal_moves() {
                    let undo = p.make_move(mv);
                    let count = perft(&mut p, depth - 1);
                    p.unmake_move(mv, undo);
                    println!("{mv}: {count}");
                    total += count;
                }
                total
            } else {
                perft(&mut p, depth)
            };
            println!(
                "nodes {nodes} time_ms {} nps {}",
                start.elapsed().as_millis(),
                (nodes as f64 / start.elapsed().as_secs_f64().max(0.000001)) as u64
            );
            Ok(())
        }
        Some("bench") => bench(&args[1..]),
        Some("help") | Some("--help") | Some("-h") => {
            println!(
                "Chen chess engine\n  chen [uci]\n  chen perft|divide DEPTH [FEN]\n  chen bench [--depth N] [--eval material|tapered] [--no-tt] [--no-qsearch] [--no-ordering] [--suite FILE] [--hash MB]\nBench emits JSON Lines, with a fresh searcher/table per position."
            );
            Ok(())
        }
        Some(command) => Err(format!("unknown command: {command}")),
    }
}
fn output(line: impl AsRef<str>) {
    println!("{}", line.as_ref());
    let _ = io::stdout().flush();
}
struct Job {
    stop: Arc<AtomicBool>,
    worker: JoinHandle<Searcher>,
}
fn finish(job: &mut Option<Job>, searcher: &mut Option<Searcher>) {
    if let Some(job) = job.take() {
        job.stop.store(true, Ordering::Relaxed);
        if let Ok(engine) = job.worker.join() {
            *searcher = Some(engine);
        }
    }
}
fn uci() -> Result<(), String> {
    let mut position = Position::startpos();
    let mut history = Vec::new();
    let mut config = SearchConfig::default();
    let mut eval_name = "tapered".to_string();
    let mut engine = Some(Searcher::new(evaluator(&eval_name)?, config.clone()));
    let mut job: Option<Job> = None;
    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(command) = tokens.first().copied() else {
            continue;
        };
        match command {
            "uci" => {
                output("id name Chen 0.1.0");
                output("id author Chen contributors");
                output("option name Hash type spin default 16 min 1 max 1024");
                output("option name Eval type combo default tapered var tapered var material");
                output("option name UseTT type check default true");
                output("option name Quiescence type check default true");
                output("option name MoveOrdering type check default true");
                output("uciok");
            }
            "isready" => output("readyok"),
            "quit" => {
                finish(&mut job, &mut engine);
                break;
            }
            "stop" => finish(&mut job, &mut engine),
            "ucinewgame" => {
                finish(&mut job, &mut engine);
                if let Some(e) = engine.as_mut() {
                    e.clear();
                }
                history.clear();
            }
            "setoption" => {
                finish(&mut job, &mut engine);
                let name = tokens.iter().position(|&t| t == "name");
                let value = tokens.iter().position(|&t| t == "value");
                if let (Some(n), Some(v)) = (name, value)
                    && n < v
                    && v + 1 < tokens.len()
                {
                    let name = tokens[n + 1..v].join(" ").to_lowercase();
                    let value = tokens[v + 1..].join(" ");
                    let mut next = config.clone();
                    let mut next_eval = eval_name.clone();
                    let valid = match name.as_str() {
                        "hash" => value
                            .parse::<usize>()
                            .ok()
                            .filter(|v| (1..=1024).contains(v))
                            .map(|v| next.hash_mb = v)
                            .is_some(),
                        "eval" if value == "material" || value == "tapered" => {
                            next_eval = value;
                            true
                        }
                        "usett" | "quiescence" | "moveordering" => {
                            if let Ok(enabled) = value.parse::<bool>() {
                                match name.as_str() {
                                    "usett" => next.tt = enabled,
                                    "quiescence" => next.quiescence = enabled,
                                    _ => next.move_ordering = enabled,
                                };
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    };
                    if valid {
                        config = next;
                        eval_name = next_eval;
                        engine = Some(Searcher::new(evaluator(&eval_name)?, config.clone()));
                    } else {
                        output("info string invalid or unsupported option");
                    }
                } else {
                    output("info string malformed setoption");
                }
            }
            "position" => {
                finish(&mut job, &mut engine);
                match parse_position(&tokens[1..]) {
                    Ok((p, h)) => {
                        position = p;
                        history = h;
                    }
                    Err(e) => output(format!("info string {e}")),
                }
            }
            "go" => {
                finish(&mut job, &mut engine);
                match parse_limits(&tokens[1..], &position) {
                    Ok(limits) => {
                        let mut searcher = engine.take().unwrap_or_else(|| {
                            Searcher::new(evaluator(&eval_name).unwrap(), config.clone())
                        });
                        let stop = searcher.stop_handle();
                        stop.store(false, Ordering::Relaxed);
                        let mut p = position;
                        let h = history.clone();
                        let worker = thread::spawn(move || {
                            let result =
                                searcher.search(&mut p, limits, &h, |r| output(r.uci_info()));
                            output(format!(
                                "bestmove {}",
                                result.best_move.map_or("0000".into(), |m| m.to_string())
                            ));
                            searcher
                        });
                        job = Some(Job { stop, worker });
                    }
                    Err(e) => output(format!("info string {e}")),
                }
            }
            "d" => output(format!("info string {}", position.to_fen())),
            _ => {}
        }
    }
    finish(&mut job, &mut engine);
    Ok(())
}
fn parse_position(tokens: &[&str]) -> Result<(Position, Vec<u64>), String> {
    let moves_at = tokens
        .iter()
        .position(|&t| t == "moves")
        .unwrap_or(tokens.len());
    let mut p = match tokens.first() {
        Some(&"startpos") if moves_at == 1 => Position::startpos(),
        Some(&"fen") if moves_at == 7 => Position::from_fen(&tokens[1..7].join(" "))?,
        _ => return Err("invalid position command".into()),
    };
    let mut history = Vec::new();
    if moves_at < tokens.len() {
        for token in &tokens[moves_at + 1..] {
            let mv = p.parse_uci_move(token)?;
            history.push(p.hash());
            p.make_move(mv);
        }
    }
    Ok((p, history))
}
fn parse_limits(tokens: &[&str], p: &Position) -> Result<SearchLimits, String> {
    let mut limits = SearchLimits::default();
    let mut remaining = None;
    let mut increment = 0;
    let mut moves_to_go = 30;
    let mut infinite = false;
    let mut explicit = false;
    let mut i = 0;
    while i < tokens.len() {
        let key = tokens[i];
        if key == "infinite" {
            infinite = true;
            i += 1;
            continue;
        }
        if !matches!(
            key,
            "depth" | "nodes" | "movetime" | "wtime" | "btime" | "winc" | "binc" | "movestogo"
        ) {
            return Err(format!("unsupported go parameter: {key}"));
        }
        let value = tokens
            .get(i + 1)
            .ok_or("missing go value")?
            .parse::<u64>()
            .map_err(|_| "invalid go value")?;
        match key {
            "depth" => {
                limits.depth = value.clamp(1, 64) as u8;
                explicit = true;
            }
            "nodes" => {
                limits.nodes = Some(value);
                explicit = true;
            }
            "movetime" => {
                limits.time = Some(Duration::from_millis(value));
                explicit = true;
            }
            "movestogo" => moves_to_go = value.max(1),
            "wtime" if p.side_to_move() == chen::Color::White => remaining = Some(value),
            "btime" if p.side_to_move() == chen::Color::Black => remaining = Some(value),
            "winc" if p.side_to_move() == chen::Color::White => increment = value,
            "binc" if p.side_to_move() == chen::Color::Black => increment = value,
            _ => {}
        }
        i += 2;
    }
    if !infinite && limits.time.is_none() {
        if let Some(time) = remaining {
            let usable = time.saturating_sub(20).max(1);
            let budget = (time / moves_to_go)
                .saturating_add(increment.saturating_mul(3) / 4)
                .min(usable)
                .max(1);
            limits.time = Some(Duration::from_millis(budget));
        } else if !explicit {
            limits.time = Some(Duration::from_secs(1));
        }
    }
    Ok(limits)
}
fn json_string(s: &str) -> String {
    let mut result = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < ' ' => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}
fn bench(args: &[String]) -> Result<(), String> {
    let mut config = SearchConfig::default();
    let mut depth = 4;
    let mut eval = "tapered".to_string();
    let mut suite = "benches/positions.txt".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-tt" => config.tt = false,
            "--no-qsearch" => config.quiescence = false,
            "--no-ordering" => config.move_ordering = false,
            "--depth" | "--eval" | "--suite" | "--hash" => {
                let value = args.get(i + 1).ok_or("missing bench value")?;
                match args[i].as_str() {
                    "--depth" => depth = value.parse::<u8>().map_err(|_| "invalid depth")?,
                    "--eval" => eval = value.clone(),
                    "--suite" => suite = value.clone(),
                    _ => config.hash_mb = value.parse().map_err(|_| "invalid hash size")?,
                };
                i += 1;
            }
            other => return Err(format!("unknown bench argument: {other}")),
        }
        i += 1;
    }
    if !(1..=64).contains(&depth) || !(1..=1024).contains(&config.hash_mb) {
        return Err("depth must be 1..64; hash must be 1..1024 MiB".into());
    }
    let contents = std::fs::read_to_string(&suite).map_err(|e| format!("{suite}: {e}"))?;
    let positions: Vec<_> = contents
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|line| {
            let (name, fen) = line.split_once('|').ok_or("suite format is name | FEN")?;
            Ok((name.trim(), Position::from_fen(fen.trim())?))
        })
        .collect::<Result<_, String>>()?;
    if positions.is_empty() {
        return Err("empty benchmark suite".into());
    }
    for (name, mut p) in positions {
        let fen = p.to_fen();
        let mut searcher = Searcher::new(evaluator(&eval)?, config.clone());
        let result = searcher.search(
            &mut p,
            SearchLimits {
                depth,
                nodes: None,
                time: None,
            },
            &[],
            |_| {},
        );
        println!(
            "{{\"engine\":\"chen\",\"version\":\"{}\",\"position\":{},\"fen\":{},\"eval\":{},\"tt\":{},\"quiescence\":{},\"ordering\":{},\"hash_mb\":{},\"requested_depth\":{},\"depth\":{},\"score_cp\":{},\"bestmove\":{},\"nodes\":{},\"qnodes\":{},\"elapsed_ns\":{},\"nps\":{},\"stopped\":{},\"pv\":{}}}",
            env!("CARGO_PKG_VERSION"),
            json_string(name),
            json_string(&fen),
            json_string(&eval),
            config.tt,
            config.quiescence,
            config.move_ordering,
            config.hash_mb,
            depth,
            result.depth,
            result.score,
            json_string(&result.best_move.map_or("0000".into(), |m| m.to_string())),
            result.nodes,
            result.qnodes,
            result.elapsed.as_nanos(),
            result.nps(),
            result.stopped,
            json_string(
                &result
                    .pv
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn position_parse_is_transactional() {
        assert!(parse_position(&["startpos", "moves", "e2e4", "e7e5"]).is_ok());
        assert!(parse_position(&["startpos", "moves", "e2e5"]).is_err());
        assert!(parse_position(&["fen", "bad"]).is_err());
    }
    #[test]
    fn clock_and_limits() {
        let p = Position::startpos();
        let l = parse_limits(&["wtime", "30000", "winc", "1000", "btime", "5"], &p).unwrap();
        assert_eq!(l.time, Some(Duration::from_millis(1750)));
        assert_eq!(parse_limits(&["depth", "3"], &p).unwrap().time, None);
        assert!(parse_limits(&["searchmoves", "e2e4"], &p).is_err());
    }
}
