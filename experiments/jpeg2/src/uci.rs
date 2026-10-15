//! Protocol history owns input transitions. Learned inference never makes a child.
use crate::{
    features::RootFacts,
    model::{Controls, Model},
};
use chen::Position;
use chen_relation_r1::{self as relation, Mode, Verdict};
use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
fn emit(line: impl AsRef<str>) {
    println!("{}", line.as_ref());
    io::stdout().flush().ok();
}
fn stopped(stop: &AtomicBool, deadline: Instant) -> bool {
    stop.load(Ordering::Relaxed) || Instant::now() >= deadline
}
fn choose(
    p: Position,
    history: Vec<u64>,
    millis: u64,
    model: Arc<Model>,
    controls: Controls,
    stop: Arc<AtomicBool>,
) {
    let started = Instant::now();
    let deadline = started + Duration::from_millis(millis.saturating_sub(20).max(1));
    let root = RootFacts::new(&p);
    let actions = root.actions();
    if actions.is_empty() {
        emit("info string terminal_root no_legal_actions");
        emit("bestmove 0000");
        return;
    }
    let encoded = started.elapsed();
    let rows: Vec<_> = actions.iter().map(|a| a.values).collect();
    let scores = model.score(&rows, &controls);
    let inferred = started.elapsed();
    let learned = scores.full_best.unwrap_or(0);
    let mut selected = learned;
    let mut mate_certificates = 0;
    let mut certificate_unknown = 0;
    let mut checks = 0;
    if !stopped(&stop, deadline) {
        let checking: Vec<_> = actions
            .iter()
            .filter(|a| root.relation.decisions()[a.decision_index].gives_check)
            .map(|a| a.mv)
            .collect();
        checks = checking.len();
        if !checking.is_empty() {
            let (base, all) = relation::codec_root_actions(&p);
            let checking: Vec<_> = all
                .into_iter()
                .filter(|a| checking.contains(&a.mv))
                .collect();
            let (labels, _) = relation::classify(&base, &checking, Mode::Stream, 4096);
            let mates: Vec<_> = labels
                .iter()
                .filter(|a| a.verdict == Verdict::Mate)
                .map(|a| a.mv)
                .collect();
            certificate_unknown = labels
                .iter()
                .filter(|a| a.verdict == Verdict::Unknown)
                .count();
            mate_certificates = mates.len();
            if !mates.is_empty() {
                selected = actions
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| mates.contains(&a.mv))
                    .max_by(|(ia, _), (ib, _)| {
                        scores.logits[*ia]
                            .total_cmp(&scores.logits[*ib])
                            .then_with(|| ib.cmp(ia))
                    })
                    .map(|(i, _)| i)
                    .unwrap();
            }
        }
    }
    let repeated = history
        .iter()
        .rev()
        .take(p.halfmove_clock() as usize)
        .filter(|&&h| h == p.hash())
        .count();
    emit(format!(
        "info string direct_model={} actions={} encode_us={} infer_us={} pair_terms={} base_choice={} learned_choice={} final_choice={} ranking_changed={} checking_actions={} mate_certificates={} certificate_unknown={} repetition_prior={} search_nodes=0 successor_positions=0 decision_source={}",
        model.name,
        actions.len(),
        encoded.as_micros(),
        (inferred - encoded).as_micros(),
        scores.interaction_terms,
        actions[scores.base_best.unwrap_or(0)].mv,
        actions[learned].mv,
        actions[selected].mv,
        scores.base_best != scores.full_best,
        checks,
        mate_certificates,
        certificate_unknown,
        repeated,
        if selected == learned {
            "learned_policy"
        } else {
            "exact_immediate_mate"
        }
    ));
    if mate_certificates > 0 {
        emit(format!(
            "info depth 0 score mate 1 time {} pv {}",
            started.elapsed().as_millis(),
            actions[selected].mv
        ));
    } else {
        emit(format!(
            "info depth 0 time {} pv {}",
            started.elapsed().as_millis(),
            actions[selected].mv
        ));
    }
    emit(format!("bestmove {}", actions[selected].mv));
}
fn parse_position(line: &str) -> Result<(Position, Vec<u64>), String> {
    let w: Vec<_> = line.split_whitespace().collect();
    let (mut p, mut i) = if w.get(1) == Some(&"startpos") {
        (Position::startpos(), 2)
    } else if w.get(1) == Some(&"fen") && w.len() >= 8 {
        (Position::from_fen(&w[2..8].join(" "))?, 8)
    } else {
        return Err("expected startpos or six-field fen".into());
    };
    let mut history = Vec::new();
    if w.get(i) == Some(&"moves") {
        i += 1;
        for text in &w[i..] {
            let mv = p.parse_uci_move(text)?;
            history.push(p.hash());
            p.make_move(mv);
        }
    }
    Ok((p, history))
}
fn movetime(line: &str, p: &Position) -> u64 {
    let w: Vec<_> = line.split_whitespace().collect();
    let value = |key: &str| {
        w.iter()
            .position(|x| *x == key)
            .and_then(|i| w.get(i + 1))
            .and_then(|s| s.parse::<u64>().ok())
    };
    if let Some(ms) = value("movetime") {
        return ms;
    }
    let white = p.side_to_move() == chen::Color::White;
    if let Some(time) = value(if white { "wtime" } else { "btime" }) {
        let inc = value(if white { "winc" } else { "binc" }).unwrap_or(0);
        return (time / value("movestogo").unwrap_or(30).max(1) + inc * 3 / 4)
            .min(time.saturating_sub(50).max(1));
    }
    1000
}
fn finish(job: &mut Option<thread::JoinHandle<()>>, stop: &AtomicBool) {
    if let Some(h) = job.take() {
        stop.store(true, Ordering::Relaxed);
        h.join().ok();
    }
}
pub fn run(model: Model, controls: Controls) -> Result<(), String> {
    if let Some(name) = &controls.omit_channel {
        if !model.groups.iter().any(|g| &g.name == name) {
            return Err(format!("unknown model channel {name}"));
        }
    }
    let model = Arc::new(model);
    let stop = Arc::new(AtomicBool::new(false));
    let mut job = None;
    let mut p = Position::startpos();
    let mut history = Vec::new();
    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        match line {
            "uci" => {
                emit("id name JPEG2 direct channel policy");
                emit("id author Chen experiments");
                emit("option name Threads type spin default 1 min 1 max 1");
                emit("uciok");
            }
            "isready" => emit("readyok"),
            "stop" => finish(&mut job, &stop),
            "quit" => {
                finish(&mut job, &stop);
                break;
            }
            "ucinewgame" => {
                finish(&mut job, &stop);
                history.clear();
            }
            _ if line.starts_with("position ") => {
                finish(&mut job, &stop);
                match parse_position(line) {
                    Ok((position, h)) => {
                        p = position;
                        history = h;
                    }
                    Err(error) => emit(format!("info string position_error {error}")),
                }
            }
            _ if line.starts_with("go") => {
                finish(&mut job, &stop);
                stop.store(false, Ordering::Relaxed);
                let time = movetime(line, &p);
                let m = model.clone();
                let s = stop.clone();
                let c = controls.clone();
                let h = history.clone();
                job = Some(thread::spawn(move || choose(p, h, time, m, c, s)));
            }
            _ => {}
        }
    }
    finish(&mut job, &stop);
    Ok(())
}
