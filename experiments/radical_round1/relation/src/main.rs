use chen::Position;
use chen_relation_r1::{
    Classified, Mode, Stats, Verdict, checking_actions, classify, codec_root_actions, mate_threats,
    orthodox_root_actions, root_actions, root_terminal,
};
use std::{hint::black_box, time::Instant};
fn scalar(p: &Position) -> Vec<Classified> {
    let mut out = Vec::new();
    for mv in p.legal_moves() {
        let mut child = *p;
        child.make_move(mv);
        let check = child.is_in_check(child.side_to_move());
        let has = child.has_legal_move();
        out.push(Classified {
            mv,
            check,
            verdict: if has {
                Verdict::HasReply
            } else if check {
                Verdict::Mate
            } else {
                Verdict::Stalemate
            },
            witness: None,
        });
    }
    out
}
fn key(v: &[Classified]) -> Vec<(String, bool, Verdict)> {
    let mut v: Vec<_> = v
        .iter()
        .map(|x| (x.mv.to_string(), x.check, x.verdict))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: relation CORPUS.tsv [REPEATS=5] [--threats] [--labels]");
        std::process::exit(2)
    }
    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let threats = args.iter().any(|x| x == "--threats");
    let labels = args.iter().any(|x| x == "--labels");
    let frontend = args
        .iter()
        .find_map(|x| x.strip_prefix("--frontend="))
        .unwrap_or("own");
    let generate = match frontend {
        "own" => root_actions,
        "codec" => codec_root_actions,
        "orthodox" => orthodox_root_actions,
        _ => panic!("unknown front end"),
    };
    let input = std::fs::read_to_string(&args[1]).unwrap();
    let mut cases = Vec::new();
    for line in input
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split('|').collect();
        if fields.len() != 4 || fields[0] == "id" {
            continue;
        }
        cases.push((
            fields[0].to_owned(),
            fields[2].to_owned(),
            Position::from_fen(fields[3]).unwrap(),
        ));
    }
    for (id, family, p) in &cases {
        let oracle = scalar(p);
        let (base, actions) = generate(p);
        let terminal = root_terminal(&base);
        assert_eq!(
            terminal,
            if p.has_legal_move() {
                Verdict::HasReply
            } else if p.is_in_check(p.side_to_move()) {
                Verdict::Mate
            } else {
                Verdict::Stalemate
            }
        );
        if labels {
            println!("ROOT\t{id}\t{terminal:?}");
        }
        for mode in [Mode::Stream, Mode::Full, Mode::Grouped] {
            let (got, _) = classify(&base, &actions, mode, usize::MAX);
            assert_eq!(
                key(&oracle),
                key(&got),
                "{id} {family} {mode:?} {}",
                p.to_fen()
            );
            for row in &got {
                if let Some(w) = row.witness {
                    let mut child = *p;
                    child.make_move(row.mv);
                    assert!(child.legal_moves().contains(&w), "invalid witness {id}");
                }
            }
            if labels {
                let list = |predicate: fn(&Classified) -> bool| {
                    got.iter()
                        .filter(|x| predicate(x))
                        .map(|x| format!("\"{}\"", x.mv))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                println!(
                    "{{\"kind\":\"classification\",\"id\":\"{id}\",\"variant\":\"{mode:?}\",\"legal\":[{}],\"checking\":[{}],\"mates\":[{}],\"stalemates\":[{}],\"unknown\":[{}]}}",
                    list(|_| true),
                    list(|x| x.check),
                    list(|x| x.verdict == Verdict::Mate),
                    list(|x| x.verdict == Verdict::Stalemate),
                    list(|x| x.verdict == Verdict::Unknown)
                );
            }
        }
        if threats {
            let threat_started = Instant::now();
            let results = mate_threats(&base, &actions, usize::MAX);
            let threat_us = threat_started.elapsed().as_secs_f64() * 1e6;
            let scalar_started = Instant::now();
            for (mv, threat, complete) in results {
                assert!(complete);
                let mut child = *p;
                child.make_move(mv);
                let expected = scalar(&child)
                    .into_iter()
                    .filter(|x| x.verdict == Verdict::Mate)
                    .map(|x| x.mv)
                    .collect::<Vec<_>>();
                assert_eq!(threat.is_some(), !expected.is_empty(), "threat {id} {mv}");
                if let Some(w) = threat {
                    assert!(expected.contains(&w));
                }
                println!(
                    "THREAT\t{id}\t{mv}\t{}\t{complete}",
                    threat.map_or_else(|| "-".into(), |x| x.to_string())
                );
            }
            let scalar_us = scalar_started.elapsed().as_secs_f64() * 1e6;
            println!(
                "{{\"kind\":\"threat_timing\",\"id\":\"{id}\",\"family\":\"{family}\",\"relation_us\":{threat_us},\"scalar_validation_us\":{scalar_us}}}"
            );
        }
        if args.iter().any(|x| x == "--budgets") {
            for budget in [0, 1, 2, 4, 8, 16, 32] {
                let (got, stats) = classify(&base, &actions, Mode::Stream, budget);
                let mut known = 0;
                let mut mates = 0;
                for row in &got {
                    if row.verdict != Verdict::Unknown {
                        known += 1;
                        let expected = oracle.iter().find(|x| x.mv == row.mv).unwrap();
                        assert_eq!(row.verdict, expected.verdict);
                    }
                    if row.verdict == Verdict::Mate {
                        mates += 1;
                    }
                }
                println!(
                    "{{\"kind\":\"budget\",\"id\":\"{id}\",\"budget\":{budget},\"known\":{known},\"mates\":{mates},\"total\":{},\"tested_templates\":{}}}",
                    actions.len(),
                    stats.tested_templates
                );
            }
        }
    }
    eprintln!("verified {} roots, all variants and witnesses", cases.len());
    let variants = if args.iter().any(|x| x == "--reverse") {
        ["grouped", "full", "stream", "scalar"]
    } else {
        ["scalar", "stream", "full", "grouped"]
    };
    for subset in ["all", "checking"] {
        for variant in variants {
            let mut total = Stats::default();
            let mut us = 0.0;
            let mut actions_total = 0;
            let mut checksum = 0usize;
            for (id, family, p) in &cases {
                let start = Instant::now();
                for _ in 0..repeats {
                    if variant == "scalar" {
                        for mv in p.legal_moves() {
                            let mut child = *p;
                            child.make_move(mv);
                            let check = child.is_in_check(child.side_to_move());
                            if subset == "checking" && !check {
                                continue;
                            }
                            actions_total += 1;
                            checksum += black_box(child.has_legal_move()) as usize;
                        }
                    } else {
                        let (base, mut actions) = generate(p);
                        if subset == "checking" {
                            actions = checking_actions(&base, &actions);
                        }
                        actions_total += actions.len();
                        let mode = match variant {
                            "stream" => Mode::Stream,
                            "full" => Mode::Full,
                            _ => Mode::Grouped,
                        };
                        let (got, stats) = classify(&base, &actions, mode, usize::MAX);
                        checksum += black_box(
                            got.iter()
                                .filter(|x| x.verdict == Verdict::HasReply)
                                .count(),
                        );
                        total.add(&stats);
                    }
                }
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                us += elapsed;
                println!(
                    "{{\"kind\":\"case_timing\",\"id\":\"{id}\",\"family\":\"{family}\",\"variant\":\"{variant}\",\"subset\":\"{subset}\",\"us_per_root\":{}}}",
                    elapsed / repeats as f64
                );
            }
            println!(
                "{{\"kind\":\"aggregate\",\"variant\":\"{variant}\",\"subset\":\"{subset}\",\"roots\":{},\"repeats\":{repeats},\"us_per_root\":{},\"actions\":{actions_total},\"checksum\":{checksum},\"templates\":{},\"tested_templates\":{},\"pseudo_pairs\":{},\"tested_pairs\":{},\"legal_pairs\":{},\"ray_steps\":{},\"words\":{},\"retained_bytes\":{},\"retired\":{},\"exhausted\":{}}}",
                cases.len(),
                us / (cases.len() * repeats) as f64,
                total.templates,
                total.tested_templates,
                total.pseudo_pairs,
                total.tested_pairs,
                total.legal_pairs,
                total.ray_steps,
                total.words,
                total.retained_bytes,
                total.retired,
                total.exhausted
            );
        }
    }
    for subset in ["all", "checking"] {
        for variant in variants {
            let mut elapsed = 0.0;
            let mut checksum = 0;
            let mut count = 0;
            for (id, family, p) in &cases {
                let (base, mut actions) = generate(p);
                if subset == "checking" {
                    actions = checking_actions(&base, &actions);
                }
                count += actions.len();
                let start = Instant::now();
                for _ in 0..repeats {
                    if variant == "scalar" {
                        for a in &actions {
                            let mut child = *p;
                            child.make_move(a.mv);
                            checksum += black_box(child.has_legal_move()) as usize;
                        }
                    } else {
                        let mode = match variant {
                            "stream" => Mode::Stream,
                            "full" => Mode::Full,
                            _ => Mode::Grouped,
                        };
                        let (got, _) = classify(&base, &actions, mode, usize::MAX);
                        checksum += black_box(
                            got.iter()
                                .filter(|x| x.verdict == Verdict::HasReply)
                                .count(),
                        );
                    }
                }
                let us = start.elapsed().as_secs_f64() * 1e6 / repeats as f64;
                elapsed += us;
                println!(
                    "{{\"kind\":\"kernel_case\",\"id\":\"{id}\",\"family\":\"{family}\",\"variant\":\"{variant}\",\"subset\":\"{subset}\",\"us_per_root\":{us}}}"
                );
            }
            println!(
                "{{\"kind\":\"kernel_aggregate\",\"variant\":\"{variant}\",\"subset\":\"{subset}\",\"us_per_root\":{},\"actions\":{count},\"checksum\":{checksum}}}",
                elapsed / cases.len() as f64
            );
        }
    }
}
