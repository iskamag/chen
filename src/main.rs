use chen::{
    OpeningBook, Position, SearchConfig, SearchLimits, Searcher,
    backend::{CpuBatchEvaluator, SearchBackend},
    codec_policy::CodecPolicy,
    core::perft,
    drowsy::{DrowsyConfig, DrowsySearch},
    eval::evaluator,
    frontier::{FrontierConfig, FrontierSearch},
    resident::{HostTree, ResidentSearch},
    selective::{HostSelectiveTree, SelectiveConfig, SelectiveSearch},
    strength::{BudgetStrength, StrengthPolicy},
};
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
    #[cfg(all(feature = "hip", target_os = "linux"))]
    let _configured_runtime = chen::neural::preload_configured_runtime()?;
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
        Some("gpu-perft") => gpu_perft(&args[1..]),
        Some("help") | Some("--help") | Some("-h") => {
            println!(
                "Chen chess engine\n  chen [uci]\n  chen perft|divide DEPTH [FEN]\n  chen gpu-perft DEPTH [--max-positions N] [FEN]\n  chen bench [--depth N] [--movetime MS] [--nodes N] [--eval material|tapered|neural] [--no-tt] [--no-qsearch] [--no-ordering] [--pvs] [--aspiration CP] [--book] [--book-seed N] [--book-plies N] [--suite FILE] [--hash MB] [--backend alphabeta|codec-policy|drowsy|cpu-frontier|hip-frontier|cpu-resident|hip-resident|cpu-selective|hip-selective|hip-bounded|hip-root-pool|hip-root-retry] [--batch-size N] [--max-positions N] [--selective-batch N] [--selective-candidates N]\nBench emits JSON Lines, with a fresh searcher/table per position."
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
fn gpu_perft(args: &[String]) -> Result<(), String> {
    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        let depth = args
            .first()
            .ok_or("usage: chen gpu-perft DEPTH [--max-positions N] [FEN]")?
            .parse::<u32>()
            .map_err(|_| "invalid depth")?;
        if depth > 10 {
            return Err("GPU perft CLI depth must be <=10".into());
        }
        let mut index = 1;
        let mut capacity = 1_000_000;
        if args.get(index).is_some_and(|s| s == "--max-positions") {
            capacity = args
                .get(index + 1)
                .ok_or("missing arena capacity")?
                .parse()
                .map_err(|_| "invalid arena capacity")?;
            index += 2;
        }
        let p = if args.len() > index {
            Position::from_fen(&args[index..].join(" "))?
        } else {
            Position::startpos()
        };
        let mut gpu = chen::device::HipRules::from_environment(capacity)?;
        let start = Instant::now();
        let (nodes, stats) = gpu
            .perft(&chen::rules::RuleState::from(&p), depth)
            .map_err(|e| e.to_string())?;
        println!(
            "{{\"backend\":\"hip-resident\",\"fen\":{},\"depth\":{},\"nodes\":{},\"max_positions\":{},\"arena_positions\":{},\"classified_positions\":{},\"kernel_ms\":{},\"elapsed_ns\":{}}}",
            json_string(&p.to_fen()),
            depth,
            nodes,
            capacity,
            stats.arena_positions,
            stats.classified_positions,
            stats.kernel_ms,
            start.elapsed().as_nanos()
        );
        Ok(())
    }
    #[cfg(not(all(feature = "hip", target_os = "linux")))]
    {
        let _ = args;
        Err("HIP requires Linux and cargo build --release --features hip".into())
    }
}
struct Job {
    stop: Arc<AtomicBool>,
    worker: JoinHandle<Box<dyn SearchBackend>>,
}
fn finish(job: &mut Option<Job>, searcher: &mut Option<Box<dyn SearchBackend>>) {
    if let Some(job) = job.take() {
        job.stop.store(true, Ordering::Relaxed);
        if let Ok(engine) = job.worker.join() {
            *searcher = Some(engine);
        }
    }
}
fn make_backend(
    name: &str,
    eval: &str,
    config: SearchConfig,
    frontier: FrontierConfig,
    selective: SelectiveConfig,
    drowsy: DrowsyConfig,
) -> Result<Box<dyn SearchBackend>, String> {
    match name {
        "alphabeta" => Ok(Box::new(Searcher::new(evaluator(eval)?, config))),
        "codec-policy" => {
            if eval != "tapered" {
                return Err(
                    "codec-policy owns tapered patch evaluation; set Eval to tapered".into(),
                );
            }
            Ok(Box::new(CodecPolicy::default()))
        }
        "drowsy" => {
            if eval != "tapered" {
                return Err("Drowsy owns the tapered evaluator; set Eval to tapered".into());
            }
            Ok(Box::new(DrowsySearch::new(drowsy)?))
        }
        "cpu-selective" => {
            if eval == "neural" {
                return Err("neural evaluator requires hip-selective".into());
            }
            Ok(Box::new(SelectiveSearch::new(
                Box::new(HostSelectiveTree::new(frontier.max_positions)?),
                eval,
                selective,
            )?))
        }
        "hip-root-retry" => {
            #[cfg(all(feature = "hip", target_os = "linux"))]
            {
                Ok(Box::new(
                    chen::hip_root_retry::HipRootRetry::from_environment(eval, config.quiescence)?,
                ))
            }
            #[cfg(not(all(feature = "hip", target_os = "linux")))]
            {
                Err("HIP requires Linux and cargo build --release --features hip".into())
            }
        }
        "hip-root-pool" => {
            #[cfg(all(feature = "hip", target_os = "linux"))]
            {
                Ok(Box::new(
                    chen::hip_root_pool::HipRootPool::from_environment(eval, config.quiescence)?,
                ))
            }
            #[cfg(not(all(feature = "hip", target_os = "linux")))]
            {
                Err("HIP requires Linux and cargo build --release --features hip".into())
            }
        }
        "hip-bounded" => {
            #[cfg(all(feature = "hip", target_os = "linux"))]
            {
                Ok(Box::new(
                    chen::hip_bounded::HipBoundedSearch::from_environment(eval, config.quiescence)?,
                ))
            }
            #[cfg(not(all(feature = "hip", target_os = "linux")))]
            {
                Err("HIP requires Linux and cargo build --release --features hip".into())
            }
        }
        "hip-selective" => {
            #[cfg(all(feature = "hip", target_os = "linux"))]
            {
                Ok(Box::new(SelectiveSearch::new(
                    if eval == "neural" {
                        Box::new(chen::neural::HipNeuralSelectiveTree::from_environment(
                            frontier.max_positions,
                        )?)
                    } else {
                        Box::new(chen::device::HipSelectiveTree::from_environment(
                            frontier.max_positions,
                        )?)
                    },
                    eval,
                    selective,
                )?))
            }
            #[cfg(not(all(feature = "hip", target_os = "linux")))]
            {
                Err("HIP requires Linux and cargo build --release --features hip".into())
            }
        }
        "cpu-frontier" => Ok(Box::new(FrontierSearch::new(
            Box::new(CpuBatchEvaluator::new(evaluator(eval)?)),
            frontier,
        )?)),
        "cpu-resident" => Ok(Box::new(ResidentSearch::new(
            Box::new(HostTree::new(frontier.max_positions).and_then(|tree| {
                if eval == "neural" {
                    Err("neural evaluator requires hip-resident".into())
                } else {
                    Ok(tree)
                }
            })?),
            eval,
        )?)),
        "hip-resident" => {
            #[cfg(all(feature = "hip", target_os = "linux"))]
            {
                Ok(Box::new(ResidentSearch::new(
                    if eval == "neural" {
                        Box::new(chen::neural::HipNeuralTree::from_environment(
                            frontier.max_positions,
                        )?)
                    } else {
                        Box::new(chen::device::HipRules::from_environment(
                            frontier.max_positions,
                        )?)
                    },
                    eval,
                )?))
            }
            #[cfg(not(all(feature = "hip", target_os = "linux")))]
            {
                Err("HIP requires Linux and cargo build --release --features hip".into())
            }
        }
        "hip-frontier" => {
            #[cfg(all(feature = "hip", target_os = "linux"))]
            {
                Ok(Box::new(FrontierSearch::new(
                    Box::new(chen::hip::HipBatchEvaluator::from_environment(
                        eval,
                        frontier.batch_size,
                    )?),
                    frontier,
                )?))
            }
            #[cfg(not(all(feature = "hip", target_os = "linux")))]
            {
                Err("HIP requires Linux and cargo build --release --features hip".into())
            }
        }
        _ => Err(format!("unknown search backend: {name}")),
    }
}
fn uci() -> Result<(), String> {
    let mut position = Position::startpos();
    let mut history = Vec::new();
    let mut config = SearchConfig::default();
    let mut eval_name = "tapered".to_string();
    let mut backend_name = "alphabeta".to_string();
    let mut frontier_config = FrontierConfig::default();
    let mut selective_config = SelectiveConfig::default();
    let mut drowsy_config = DrowsyConfig::default();
    let mut level = 20;
    let mut book: Option<Arc<OpeningBook>> = None;
    let mut book_seed = 0u64;
    let mut book_plies = 12usize;
    let mut engine = Some(make_backend(
        &backend_name,
        &eval_name,
        config.clone(),
        frontier_config.clone(),
        selective_config,
        drowsy_config,
    )?);
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
                output(
                    "option name Eval type combo default tapered var tapered var material var neural",
                );
                output("option name UseTT type check default true");
                output("option name Quiescence type check default true");
                output("option name MoveOrdering type check default true");
                output("option name PVS type check default false");
                output("option name AspirationWindow type spin default 0 min 0 max 32000");
                output(
                    "option name Backend type combo default alphabeta var alphabeta var codec-policy var drowsy var cpu-frontier var hip-frontier var cpu-resident var hip-resident var cpu-selective var hip-selective var hip-bounded var hip-root-pool var hip-root-retry",
                );
                output("option name DrowsyDelta type spin default 80 min 0 max 32000");
                output("option name DrowsyTheta type string default 0.55");
                output("option name SelectiveBatch type spin default 32 min 1 max 256");
                output("option name SelectiveCandidates type spin default 1024 min 1 max 1024");
                output("option name BatchSize type spin default 4096 min 1 max 65536");
                output("option name MaxPositions type spin default 250000 min 1 max 2000000");
                output("option name Level type spin default 20 min 1 max 20");
                output("option name Book type check default false");
                output("option name BookSeed type spin default 0 min 0 max 4294967295");
                output("option name BookPlies type spin default 12 min 1 max 40");
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
                    let mut next_backend = backend_name.clone();
                    let mut next_frontier = frontier_config.clone();
                    let mut next_selective = selective_config;
                    let mut next_drowsy = drowsy_config;
                    let mut next_level = level;
                    let mut next_book = book.clone();
                    let mut next_book_seed = book_seed;
                    let mut next_book_plies = book_plies;
                    let valid = match name.as_str() {
                        "book" => value
                            .parse::<bool>()
                            .ok()
                            .map(|enabled| {
                                next_book = if enabled {
                                    book.clone()
                                        .or_else(|| OpeningBook::bundled().map(Arc::new))
                                } else {
                                    None
                                };
                                if enabled && next_book.is_none() {
                                    output("info string opening book unavailable in this build");
                                }
                            })
                            .is_some(),
                        "bookseed" => value
                            .parse::<u64>()
                            .ok()
                            .map(|v| next_book_seed = v)
                            .is_some(),
                        "bookplies" => value
                            .parse::<usize>()
                            .ok()
                            .filter(|v| (1..=40).contains(v))
                            .map(|v| next_book_plies = v)
                            .is_some(),
                        "level" => value
                            .parse::<u8>()
                            .ok()
                            .filter(|v| (1..=20).contains(v))
                            .map(|v| next_level = v)
                            .is_some(),
                        "hash" => value
                            .parse::<usize>()
                            .ok()
                            .filter(|v| (1..=1024).contains(v))
                            .map(|v| next.hash_mb = v)
                            .is_some(),
                        "eval" if matches!(value.as_str(), "material" | "tapered" | "neural") => {
                            next_eval = value;
                            true
                        }
                        "backend"
                            if matches!(
                                value.as_str(),
                                "alphabeta"
                                    | "codec-policy"
                                    | "drowsy"
                                    | "cpu-frontier"
                                    | "hip-frontier"
                                    | "cpu-resident"
                                    | "hip-resident"
                                    | "cpu-selective"
                                    | "hip-selective"
                                    | "hip-bounded"
                                    | "hip-root-pool"
                                    | "hip-root-retry"
                            ) =>
                        {
                            next_backend = value;
                            true
                        }
                        "drowsydelta" => value
                            .parse::<i32>()
                            .ok()
                            .filter(|v| (0..=32000).contains(v))
                            .map(|v| next_drowsy.delta_cp = v)
                            .is_some(),
                        "drowsytheta" => value
                            .parse::<f32>()
                            .ok()
                            .filter(|v| !v.is_nan() && *v >= 0.0)
                            .map(|v| next_drowsy.theta = v)
                            .is_some(),
                        "selectivebatch" => value
                            .parse::<u32>()
                            .ok()
                            .filter(|v| {
                                (1..=256).contains(v) && *v <= next_selective.candidate_limit
                            })
                            .map(|v| next_selective.batch_leaves = v)
                            .is_some(),
                        "selectivecandidates" => value
                            .parse::<u32>()
                            .ok()
                            .filter(|v| (next_selective.batch_leaves..=1024).contains(v))
                            .map(|v| next_selective.candidate_limit = v)
                            .is_some(),
                        "batchsize" => value
                            .parse::<usize>()
                            .ok()
                            .filter(|v| (1..=65536).contains(v))
                            .map(|v| next_frontier.batch_size = v)
                            .is_some(),
                        "maxpositions" => value
                            .parse::<usize>()
                            .ok()
                            .filter(|v| (1..=2_000_000).contains(v))
                            .map(|v| next_frontier.max_positions = v)
                            .is_some(),
                        "aspirationwindow" => value
                            .parse::<u16>()
                            .ok()
                            .filter(|v| *v <= 32000)
                            .map(|v| next.aspiration_window = v)
                            .is_some(),
                        "usett" | "quiescence" | "moveordering" | "pvs" => {
                            if let Ok(enabled) = value.parse::<bool>() {
                                match name.as_str() {
                                    "usett" => next.tt = enabled,
                                    "quiescence" => next.quiescence = enabled,
                                    "pvs" => next.pvs = enabled,
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
                        match make_backend(
                            &next_backend,
                            &next_eval,
                            next.clone(),
                            next_frontier.clone(),
                            next_selective,
                            next_drowsy,
                        ) {
                            Ok(mut created) => {
                                config = next;
                                eval_name = next_eval;
                                backend_name = next_backend;
                                frontier_config = next_frontier;
                                selective_config = next_selective;
                                drowsy_config = next_drowsy;
                                level = next_level;
                                book = next_book;
                                book_seed = next_book_seed;
                                book_plies = next_book_plies;
                                created.set_book(book.clone(), book_seed, book_plies);
                                engine = Some(created);
                            }
                            Err(error) => {
                                config = next;
                                eval_name = next_eval;
                                backend_name = next_backend;
                                frontier_config = next_frontier;
                                selective_config = next_selective;
                                drowsy_config = next_drowsy;
                                level = next_level;
                                book = next_book;
                                book_seed = next_book_seed;
                                book_plies = next_book_plies;
                                engine = None;
                                output(format!("info string backend creation failed: {error}"));
                            }
                        }
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
                        let limits = BudgetStrength::new(level)?.apply(limits);
                        let mut searcher = match engine.take() {
                            Some(searcher) => searcher,
                            None => make_backend(
                                &backend_name,
                                &eval_name,
                                config.clone(),
                                frontier_config.clone(),
                                selective_config,
                                drowsy_config,
                            )?,
                        };
                        searcher.set_book(book.clone(), book_seed, book_plies);
                        let stop = searcher.stop_handle();
                        stop.store(false, Ordering::Relaxed);
                        let mut p = position;
                        let h = history.clone();
                        let worker = thread::spawn(move || {
                            let result =
                                searcher.search(&mut p, limits, &h, &mut |r| output(r.uci_info()));
                            match result {
                                Ok(result) => {
                                    #[cfg(feature = "hip")]
                                    if let Some(stats) = searcher.bounded_stats() {
                                        output(format!("info string bounded {}", stats.to_json()));
                                    }
                                    if let Some(stats) = searcher.codec_stats() {
                                        output(format!(
                                            "info string codec-policy score-kind {} alternatives {} legal {} scored {} lane-words {} counterfactual-boards 0 tree-nodes 0",
                                            stats.score_kind,
                                            stats.alternatives,
                                            stats.legal_alternatives,
                                            stats.scored_alternatives,
                                            stats.lane_words
                                        ));
                                    }
                                    if let Some(stats) = searcher.drowsy_stats() {
                                        output(format!(
                                            "info string drowsy fail-meow {} fail-zoomies {} ignored-boring {} noticed-drift {} explored-ply {} walkers {}",
                                            stats.fail_meow,
                                            stats.fail_zoomies,
                                            stats.ignored_boring,
                                            stats.noticed_drift,
                                            stats.deepest_ply,
                                            stats.peak_walkers
                                        ));
                                    }
                                    if let Some(stats) = searcher.root_pool_stats() {
                                        output(format!("info string root_pool {stats}"));
                                    }
                                    output(format!(
                                        "bestmove {}",
                                        result.best_move.map_or("0000".into(), |m| m.to_string())
                                    ));
                                }
                                Err(error) => {
                                    output(format!("info string search backend failed: {error}"));
                                    output("bestmove 0000");
                                }
                            }
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
    let mut backend_name = "alphabeta".to_string();
    let mut frontier_config = FrontierConfig::default();
    let mut selective_config = SelectiveConfig::default();
    let mut drowsy_config = DrowsyConfig::default();
    let mut depth = 4;
    let mut node_limit = None;
    let mut time_limit = None;
    let mut eval = "tapered".to_string();
    let mut suite = "benches/positions.txt".to_string();
    let mut book = false;
    let mut book_seed = 0u64;
    let mut book_plies = 12usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-tt" => config.tt = false,
            "--no-qsearch" => config.quiescence = false,
            "--no-ordering" => config.move_ordering = false,
            "--pvs" => config.pvs = true,
            "--book" => book = true,
            "--depth"
            | "--nodes"
            | "--movetime"
            | "--eval"
            | "--suite"
            | "--hash"
            | "--backend"
            | "--book-seed"
            | "--book-plies"
            | "--batch-size"
            | "--max-positions"
            | "--selective-batch"
            | "--aspiration"
            | "--selective-candidates"
            | "--drowsy-delta"
            | "--drowsy-theta" => {
                let value = args.get(i + 1).ok_or("missing bench value")?;
                match args[i].as_str() {
                    "--depth" => depth = value.parse::<u8>().map_err(|_| "invalid depth")?,
                    "--book-seed" => book_seed = value.parse().map_err(|_| "invalid book seed")?,
                    "--book-plies" => {
                        book_plies = value.parse().map_err(|_| "invalid book plies")?
                    }
                    "--aspiration" => {
                        config.aspiration_window = value
                            .parse::<u16>()
                            .map_err(|_| "invalid aspiration half-window")?
                    }
                    "--movetime" => {
                        time_limit = Some(std::time::Duration::from_millis(
                            value.parse::<u64>().map_err(|_| "invalid movetime")?,
                        ))
                    }
                    "--nodes" => {
                        node_limit = Some(value.parse::<u64>().map_err(|_| "invalid nodes")?)
                    }
                    "--drowsy-delta" => {
                        drowsy_config.delta_cp =
                            value.parse().map_err(|_| "invalid Drowsy delta")?
                    }
                    "--drowsy-theta" => {
                        drowsy_config.theta = value.parse().map_err(|_| "invalid Drowsy theta")?
                    }
                    "--eval" => eval = value.clone(),
                    "--backend" => backend_name = value.clone(),
                    "--batch-size" => {
                        frontier_config.batch_size =
                            value.parse().map_err(|_| "invalid batch size")?
                    }
                    "--selective-batch" => {
                        selective_config.batch_leaves =
                            value.parse().map_err(|_| "invalid selective batch")?
                    }
                    "--selective-candidates" => {
                        selective_config.candidate_limit =
                            value.parse().map_err(|_| "invalid selective candidates")?
                    }
                    "--max-positions" => {
                        frontier_config.max_positions =
                            value.parse().map_err(|_| "invalid frontier capacity")?
                    }
                    "--suite" => suite = value.clone(),
                    _ => config.hash_mb = value.parse().map_err(|_| "invalid hash size")?,
                };
                i += 1;
            }
            other => return Err(format!("unknown bench argument: {other}")),
        }
        i += 1;
    }
    if !(1..=64).contains(&depth)
        || !(1..=1024).contains(&config.hash_mb)
        || config.aspiration_window > 32000
    {
        return Err(
            "depth must be 1..64; hash must be 1..1024 MiB; aspiration must be 0..32000 cp".into(),
        );
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
    // The book is parsed once and shared across the suite.
    let book = book.then(|| OpeningBook::bundled().map(Arc::new)).flatten();
    if book_seed != 0 && book.is_none() {
        return Err("--book-seed needs an available opening book".into());
    }
    for (name, mut p) in positions {
        let fen = p.to_fen();
        let mut searcher = make_backend(
            &backend_name,
            &eval,
            config.clone(),
            frontier_config.clone(),
            selective_config,
            drowsy_config,
        )?;
        searcher.set_book(book.clone(), book_seed, book_plies);
        let result = searcher.search(
            &mut p,
            SearchLimits {
                depth,
                nodes: node_limit,
                time: time_limit,
            },
            &[],
            &mut |_| {},
        )?;
        let telemetry = searcher.batch_telemetry();
        let evaluator_artifact = searcher
            .evaluator_artifact_sha256()
            .map(json_string)
            .unwrap_or_else(|| "null".to_owned());
        let execution = searcher.neural_execution_stats()?.map_or_else(|| "null".to_owned(), |s| format!(
            "{{\"mode\":{},\"max_cached_shapes\":{},\"retained_budget_bytes\":{},\"first_used\":{},\"cached_shapes\":{},\"hits\":{},\"misses\":{},\"captures\":{},\"evictions\":{},\"retained_pool_bytes\":{},\"staging_logical_bytes\":{},\"peak_observed_pool_bytes\":{},\"capture_wall_ms\":{}}}",
            json_string(s.mode), s.max_cached_shapes, s.retained_budget_bytes, s.first_used, s.cached_shapes, s.hits, s.misses, s.captures, s.evictions, s.retained_pool_bytes, s.staging_logical_bytes, s.peak_observed_pool_bytes, s.capture_wall_ms
        ));
        #[cfg(feature = "hip")]
        let bounded = searcher
            .bounded_stats()
            .map_or_else(|| "null".to_owned(), |s| s.to_json());
        #[cfg(not(feature = "hip"))]
        let bounded = "null";
        let codec = searcher.codec_stats().map_or_else(|| "null".to_owned(), |s| format!(
            "{{\"alternatives\":{},\"legal_alternatives\":{},\"scored_alternatives\":{},\"lane_words\":{},\"boundary_compositions\":{},\"patch_edits\":{},\"attack_bytes\":{},\"counterfactual_boards\":0,\"tree_nodes\":0,\"score_kind\":{}}}",
            s.alternatives, s.legal_alternatives, s.scored_alternatives, s.lane_words,
            s.boundary_compositions, s.patch_edits, s.attack_bytes, json_string(s.score_kind)
        ));
        let drowsy = searcher.drowsy_stats().map_or_else(|| "null".to_owned(), |s| {
            let theta = if drowsy_config.theta.is_infinite() { json_string("inf") }
                else { drowsy_config.theta.to_string() };
            format!("{{\"delta_cp\":{},\"theta\":{},\"seed\":{},\"walkers\":{},\"max_plies\":{},\"fail_meow\":{},\"fail_zoomies\":{},\"ignored_boring\":{},\"noticed_drift\":{},\"deepest_ply\":{},\"peak_walkers\":{}}}",
                drowsy_config.delta_cp, theta, drowsy_config.seed, drowsy_config.walkers,
                drowsy_config.max_plies, s.fail_meow, s.fail_zoomies, s.ignored_boring,
                s.noticed_drift, s.deepest_ply, s.peak_walkers)
        });
        let root_pool = searcher.root_pool_stats().unwrap_or_else(|| "null".into());
        let selective = result.selective.map_or_else(|| "null".to_owned(), |s| format!(
            "{{\"abi_version\":{},\"struct_size\":{},\"batch_leaves\":{},\"candidate_limit\":{},\"arena_positions\":{},\"classified_positions\":{},\"evaluated_positions\":{},\"completed_steps\":{},\"expanded_leaves\":{},\"open_frontier_leaves\":{},\"last_selected\":{},\"last_added\":{},\"kernel_ms\":{},\"minimum_frontier_depth\":{},\"maximum_frontier_depth\":{},\"pv_depth\":{},\"root_flags\":{},\"stop_reason\":{}}}",
            s.abi_version, s.struct_size, selective_config.batch_leaves, selective_config.candidate_limit,
            s.arena_positions, s.classified_positions, s.evaluated_positions, s.completed_steps,
            s.expanded_leaves, s.open_frontier_leaves, s.last_selected, s.last_added, s.kernel_ms,
            s.minimum_frontier_depth, s.maximum_frontier_depth, s.pv_depth, s.root_flags, s.stop_reason,
        ));
        println!(
            "{{\"engine\":\"chen\",\"version\":\"{}\",\"backend\":{},\"batch_size\":{},\"max_positions\":{},\"evaluated_positions\":{},\"h2d_ms\":{},\"kernel_ms\":{},\"d2h_ms\":{},\"position\":{},\"fen\":{},\"eval\":{},\"evaluator_artifact_sha256\":{},\"neural_execution\":{},\"selective\":{},\"bounded\":{},\"root_pool\":{},\"drowsy\":{},\"codec\":{},\"book\":{},\"book_seed\":{},\"book_plies\":{},\"opening\":{},\"requested_movetime_ms\":{},\"requested_nodes\":{},\"tt\":{},\"quiescence\":{},\"ordering\":{},\"pvs\":{},\"aspiration_window\":{},\"hash_mb\":{},\"requested_depth\":{},\"depth\":{},\"score_cp\":{},\"bestmove\":{},\"nodes\":{},\"qnodes\":{},\"elapsed_ns\":{},\"nps\":{},\"stopped\":{},\"pv\":{}}}",
            env!("CARGO_PKG_VERSION"),
            json_string(&backend_name),
            frontier_config.batch_size,
            frontier_config.max_positions,
            telemetry.positions,
            telemetry.h2d_ms,
            telemetry.kernel_ms,
            telemetry.d2h_ms,
            json_string(name),
            json_string(&fen),
            json_string(&eval),
            evaluator_artifact,
            execution,
            selective,
            bounded,
            root_pool,
            drowsy,
            codec,
            result.book_move.is_some(),
            book_seed,
            book_plies,
            result
                .book_move
                .as_ref()
                .map_or_else(|| "null".to_owned(), |entry| json_string(&entry.name)),
            time_limit.map_or_else(|| "null".to_owned(), |t| t.as_millis().to_string()),
            node_limit.map_or_else(|| "null".to_owned(), |n| n.to_string()),
            config.tt && backend_name == "alphabeta",
            config.quiescence
                && matches!(
                    backend_name.as_str(),
                    "alphabeta" | "hip-bounded" | "hip-root-pool" | "hip-root-retry"
                ),
            config.move_ordering && backend_name == "alphabeta",
            config.pvs && backend_name == "alphabeta",
            if backend_name == "alphabeta" {
                config.aspiration_window
            } else {
                0
            },
            if backend_name == "alphabeta" {
                config.hash_mb
            } else {
                0
            },
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
