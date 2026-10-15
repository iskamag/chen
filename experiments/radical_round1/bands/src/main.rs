use chen::{Color, Piece, Position};
use chen_radical_bands::*;
use serde_json::{Value, json};
use std::{collections::HashSet, hint::black_box, time::Instant};

struct Row {
    id: String,
    split: String,
    family: String,
    p: Position,
}

fn parse_corpus(path: &str) -> Vec<Row> {
    std::fs::read_to_string(path)
        .expect("read corpus")
        .lines()
        .filter_map(|l| {
            if l.trim().is_empty() || l.starts_with('#') || l.starts_with("id|") {
                return None;
            }
            let x: Vec<_> = l.splitn(4, '|').collect();
            assert_eq!(x.len(), 4, "id|split|family|fen");
            Some(Row {
                id: x[0].into(),
                split: x[1].into(),
                family: x[2].into(),
                p: Position::from_fen(x[3]).unwrap(),
            })
        })
        .collect()
}

fn oracle_obligations(p: &Position, color: Color) -> Obligations {
    let king = p.king_square(color);
    let mut checkers = 0;
    let mut pinned = 0;
    let mut evasion = !0;
    let kf = (king % 8) as i8;
    let kr = (king / 8) as i8;
    for (df, dr) in [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    ] {
        let mut f = kf + df;
        let mut r = kr + dr;
        let mut friend = None;
        let mut path = 0;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let sq = (r * 8 + f) as u8;
            path |= 1u64 << sq;
            if let Some((c, pc)) = p.piece_at(sq) {
                if c == color {
                    if friend.is_some() {
                        break;
                    }
                    friend = Some(sq);
                } else {
                    if pc == Piece::Queen
                        || if df == 0 || dr == 0 {
                            pc == Piece::Rook
                        } else {
                            pc == Piece::Bishop
                        }
                    {
                        if let Some(pin) = friend {
                            pinned |= 1u64 << pin;
                        } else {
                            checkers |= 1u64 << sq;
                            evasion &= path;
                        }
                    }
                    break;
                }
            }
            f += df;
            r += dr;
        }
    }
    for sq in 0..64 {
        let Some((c, pc)) = p.piece_at(sq) else {
            continue;
        };
        if c == color {
            continue;
        }
        let df = kf - (sq % 8) as i8;
        let dr = kr - (sq / 8) as i8;
        let attacks = match pc {
            Piece::Pawn => df.abs() == 1 && dr == if c == Color::White { 1 } else { -1 },
            Piece::Knight => (df.abs() == 1 && dr.abs() == 2) || (df.abs() == 2 && dr.abs() == 1),
            Piece::King => df.abs().max(dr.abs()) == 1,
            _ => false,
        };
        if attacks {
            checkers |= 1u64 << sq;
            evasion &= 1u64 << sq;
        }
    }
    Obligations {
        checkers,
        pinned,
        evasion,
    }
}

fn bench<F: FnMut(usize) -> u64>(n: usize, iters: usize, mut f: F) -> f64 {
    let mut checksum = 0;
    // Small fixture families otherwise time only a few microseconds, and the
    // first variant can run before this CPU exits an idle frequency state.
    // Every workload gets its own warmup and >=15ms timed region. Check the
    // clock once per 32 sweeps rather than adding a timer to each query.
    let warmup = Instant::now();
    while warmup.elapsed().as_millis() < 3 {
        for _ in 0..32 {
            for j in 0..n {
                checksum ^= black_box(f(black_box(j)));
            }
        }
    }
    let start = Instant::now();
    let mut sweeps = 0usize;
    loop {
        for _ in 0..32 {
            for j in 0..n {
                checksum ^= black_box(f(black_box(j)));
            }
        }
        sweeps += 32;
        if sweeps >= iters && start.elapsed().as_millis() >= 15 {
            break;
        }
    }
    black_box(checksum);
    start.elapsed().as_nanos() as f64 / (n * sweeps) as f64
}

fn emit(out: &mut Vec<Value>, value: Value) {
    println!("{}", value);
    out.push(value);
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let arg = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let corpus = arg("--corpus").expect("--corpus id|split|family|fen");
    let iters: usize = arg("--iters").unwrap_or("200".into()).parse().unwrap();
    let rows = parse_corpus(&corpus);
    assert!(!rows.is_empty());
    let t = Tables::new();
    let states: Vec<_> = rows.iter().map(|r| Encoded::new(&r.p, &t)).collect();
    let pyramids: Vec<_> = rows.iter().map(|r| Pyramid::new(&r.p, &t)).collect();
    let avx = avx_available();
    if args.iter().any(|a| a == "--answers-only") {
        for ((r, s), pyramid) in rows.iter().zip(&states).zip(&pyramids) {
            for (variant, micro, simd) in [
                ("micro-scalar", true, false),
                ("bands-scalar", false, false),
                ("micro-avx2", true, true),
                ("bands-avx2", false, true),
            ] {
                if simd && !avx {
                    continue;
                }
                let answer = attacks(s, &t, micro, simd);
                let w = obligations(s, &t, 0);
                let b = obligations(s, &t, 1);
                println!(
                    "{}",
                    json!({"id":r.id,"fen":r.p.to_fen(),"variant":variant,"white":answer[0],"black":answer[1],"white_checkers":w.checkers,"black_checkers":b.checkers,"white_pinned":w.pinned,"black_pinned":b.pinned,"white_evasion":w.evasion,"black_evasion":b.evasion})
                );
            }
            for simd in [false, true] {
                if simd && !avx {
                    continue;
                }
                let answer = pyramid.attacks(&t, simd);
                println!(
                    "{}",
                    json!({"id":r.id,"fen":r.p.to_fen(),"variant":if simd {"pyramid-avx2"} else {"pyramid-scalar"},"white":answer[0],"black":answer[1]})
                );
            }
        }
        return;
    }
    let mut output = Vec::new();
    let unique: HashSet<_> = t.micro.iter().copied().collect();
    let ray_unique: HashSet<_> = t.micro.iter().map(|x| x >> 72).collect();
    let attack_unique: HashSet<_> = t.micro.iter().map(|x| x & ((1u128 << 72) - 1)).collect();
    let (line_classes, target_classes) = t.operator_classes();
    emit(
        &mut output,
        json!({"kind":"architecture","variant":"bands","positions":rows.len(),"avx2_available":avx,"encoded_bytes":std::mem::size_of::<Encoded>(),"pyramid_bytes":std::mem::size_of::<Pyramid>(),"lut_payload_bytes":t.bytes(),"micro_entries":t.micro.len(),"micro_combined_attack_projection_classes":unique.len(),"micro_ray_classes":ray_unique.len(),"micro_leaper_classes":attack_unique.len(),"full_line_attack_input_classes":65536,"full_line_attack_output_classes":line_classes,"king_target_obligation_input_classes":78125,"king_target_obligation_output_classes_by_king_file":target_classes,"line_count":t.line_count(),"strength":"N/A: geometry and constraint kernel, no move policy","target_cpu":std::env::var("CHEN_BUILD_TARGET").unwrap_or("unspecified".into())}),
    );
    let mut children = Vec::new();
    let mut edge_counts = [0usize; 5];
    let mut dirty_macro_counts = [0usize; 5];
    for ((r, s), pyramid) in rows.iter().zip(&states).zip(&pyramids) {
        let expect = walk_oracle(&r.p);
        assert_eq!(union_control(&r.p), expect, "union {}", r.id);
        assert_eq!(pyramid.attacks(&t, false), expect, "pyramid {}", r.id);
        if avx {
            assert_eq!(pyramid.attacks(&t, true), expect, "pyramid avx {}", r.id);
        }
        for micro in [false, true] {
            for simd in [false, true] {
                if simd && !avx {
                    continue;
                }
                assert_eq!(
                    attacks(s, &t, micro, simd),
                    expect,
                    "attacks {} micro {micro} simd {simd}",
                    r.id
                );
            }
        }
        let mut obligation_output = Vec::new();
        for color in [Color::White, Color::Black] {
            let o = obligations(s, &t, color.index());
            assert_eq!(
                o,
                oracle_obligations(&r.p, color),
                "obligations {} {:?}",
                r.id,
                color
            );
            assert_eq!(
                obligations_progressive(s, &t, color.index()),
                o,
                "progressive {} {color:?}",
                r.id
            );
            assert_eq!(
                obligations_control(&r.p, &t, color),
                o,
                "strong control {} {color:?}",
                r.id
            );
            obligation_output.push(json!({"checkers":format!("{:016x}",o.checkers),"pinned":format!("{:016x}",o.pinned),"evasion":format!("{:016x}",o.evasion),"coarse_needs_slider_refinement":needs_slider_refinement(s,&t,color.index())}));
        }
        output.push(json!({"kind":"position","id":r.id,"split":r.split,"family":r.family,"attacks":[format!("{:016x}",expect[0]),format!("{:016x}",expect[1])],"obligations":obligation_output}));
        // Verify every legal transition, including castling, EP and promotions
        // present in the corpus. Geometry is checked after actual rule updates;
        // no move legality is claimed for the descriptor itself.
        for mv in r.p.legal_moves() {
            let mut after = r.p;
            after.make_move(mv);
            let patch = edits(&r.p, &after);
            edge_counts[patch.len()] += 1;
            let mut patched = s.clone();
            patched.patch(&patch, &t);
            let mut patched_pyramid = pyramid.clone();
            dirty_macro_counts[patched_pyramid.patch(&patch, &t)] += 1;
            let rebuilt = Encoded::new(&after, &t);
            assert_eq!(patched.lines, rebuilt.lines, "patch lines {} {mv}", r.id);
            assert_eq!(patched.micro, rebuilt.micro, "patch micro {} {mv}", r.id);
            assert_eq!(patched.planes, rebuilt.planes, "patch planes {} {mv}", r.id);
            assert_eq!(patched.kings, rebuilt.kings, "patch kings {} {mv}", r.id);
            let truth = walk_oracle(&after);
            assert_eq!(
                patched_pyramid.attacks(&t, false),
                truth,
                "pyramid child {} {mv}",
                r.id
            );
            assert_eq!(
                attacks(&patched, &t, true, false),
                truth,
                "micro child {} {mv}",
                r.id
            );
            assert_eq!(
                attacks(&patched, &t, false, false),
                truth,
                "band child {} {mv}",
                r.id
            );
            if avx {
                assert_eq!(
                    attacks(&patched, &t, false, true),
                    truth,
                    "avx child {} {mv}",
                    r.id
                );
            }
            for c in [Color::White, Color::Black] {
                let expect = oracle_obligations(&after, c);
                assert_eq!(
                    obligations(&patched, &t, c.index()),
                    expect,
                    "child obligations {} {mv} {c:?}",
                    r.id
                );
                assert_eq!(
                    obligations_progressive(&patched, &t, c.index()),
                    expect,
                    "child progressive {} {mv} {c:?}",
                    r.id
                );
                assert_eq!(
                    obligations_control(&after, &t, c),
                    expect,
                    "child strong control {} {mv} {c:?}",
                    r.id
                );
            }
            children.push((s.clone(), patch, after, pyramid.clone()));
        }
    }
    emit(
        &mut output,
        json!({"kind":"exactness","root_positions":rows.len(),"all_masks_match":true,"all_obligations_match":true,"legal_transitions":children.len(),"transition_edit_counts":edge_counts,"dirty_macro_counts":dirty_macro_counts,"oracle":"independent piece/ray walker plus independent king-ray constraints; external python-chess replay requested"}),
    );
    let families: HashSet<_> = rows.iter().map(|r| r.family.clone()).collect();
    let mut families: Vec<_> = families.into_iter().collect();
    families.sort();
    if args.iter().any(|a| a == "--all-only") {
        families.clear();
    }
    families.insert(0, "ALL".into());
    for family in families {
        let indices: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| family == "ALL" || r.family == family)
            .map(|(i, _)| i)
            .collect();
        let n = indices.len();
        let mut variants = vec![
            ("union-control", 0),
            ("micro-scalar", 1),
            ("bands-scalar", 2),
            ("pyramid-scalar", 5),
        ];
        if avx {
            variants.extend([("micro-avx2", 3), ("bands-avx2", 4), ("pyramid-avx2", 6)]);
        }
        for (name, v) in variants {
            let query = bench(n, iters, |j| {
                let i = indices[j];
                let a = if v == 0 {
                    union_control(&rows[i].p)
                } else if v >= 5 {
                    pyramids[i].attacks(&t, v == 6)
                } else {
                    attacks(&states[i], &t, v == 1 || v == 3, v >= 3)
                };
                a[0] ^ a[1]
            });
            let build_query = bench(n, iters, |j| {
                let i = indices[j];
                let a = if v == 0 {
                    union_control(&rows[i].p)
                } else if v >= 5 {
                    Pyramid::new(black_box(&rows[i].p), &t).attacks(&t, v == 6)
                } else {
                    let s = Encoded::new(black_box(&rows[i].p), &t);
                    attacks(&s, &t, v == 1 || v == 3, v >= 3)
                };
                a[0] ^ a[1]
            });
            let packet = bench(n, iters, |j| {
                let i = indices[j];
                let a = if v == 0 {
                    union_control(&rows[i].p)
                } else if v >= 5 {
                    pyramids[i].attacks(&t, v == 6)
                } else {
                    attacks(&states[i], &t, v == 1 || v == 3, v >= 3)
                };
                let (w, b) = if v == 0 {
                    (
                        obligations_control(&rows[i].p, &t, Color::White),
                        obligations_control(&rows[i].p, &t, Color::Black),
                    )
                } else {
                    (
                        obligations_progressive(&states[i], &t, 0),
                        obligations_progressive(&states[i], &t, 1),
                    )
                };
                a[0] ^ a[1] ^ w.checkers ^ w.pinned ^ w.evasion ^ b.checkers ^ b.pinned ^ b.evasion
            });
            emit(
                &mut output,
                json!({"kind":"benchmark","family":family,"variant":name,"positions":n,"iters":iters,"attacks_query_ns":query,"attacks_build_query_ns":build_query,"complete_packet_ns":packet}),
            );
        }
        let o = bench(n, iters, |j| {
            let i = indices[j];
            let a = obligations(&states[i], &t, 0);
            let b = obligations(&states[i], &t, 1);
            a.checkers ^ a.pinned ^ a.evasion ^ b.checkers ^ b.pinned ^ b.evasion
        });
        let progressive = bench(n, iters, |j| {
            let i = indices[j];
            let a = obligations_progressive(&states[i], &t, 0);
            let b = obligations_progressive(&states[i], &t, 1);
            a.checkers ^ a.pinned ^ a.evasion ^ b.checkers ^ b.pinned ^ b.evasion
        });
        let control = bench(n, iters, |j| {
            let i = indices[j];
            let a = obligations_control(&rows[i].p, &t, Color::White);
            let b = obligations_control(&rows[i].p, &t, Color::Black);
            a.checkers ^ a.pinned ^ a.evasion ^ b.checkers ^ b.pinned ^ b.evasion
        });
        let coarse = indices
            .iter()
            .filter(|&&i| needs_slider_refinement(&states[i], &t, 0))
            .count()
            + indices
                .iter()
                .filter(|&&i| needs_slider_refinement(&states[i], &t, 1))
                .count();
        let nontrivial = indices
            .iter()
            .map(|&i| {
                (0..2)
                    .filter(|&c| {
                        let o = obligations(&states[i], &t, c);
                        o.pinned != 0
                            || o.checkers
                                & (states[i].planes[1 - c][2]
                                    | states[i].planes[1 - c][3]
                                    | states[i].planes[1 - c][4])
                                != 0
                    })
                    .count()
            })
            .sum::<usize>();
        emit(
            &mut output,
            json!({"kind":"constraints","family":family,"query_both_kings_ns":o,"progressive_both_kings_ns":progressive,"control_both_kings_ns":control,"coarse_refinements":coarse,"refined_nontrivial_slider_constraints":nontrivial,"queries":2*n}),
        );
    }
    let query_iters = iters.min(40);
    let patch_ns = bench(children.len(), query_iters, |i| {
        let (before, edits, _, _) = &children[i];
        let mut after = before.clone();
        after.patch(black_box(edits), &t);
        black_box(after.lines[0] as u64 ^ after.micro[0] as u64)
    });
    let patch_query_ns = bench(children.len(), query_iters, |i| {
        let (before, edits, _, _) = &children[i];
        let mut after = before.clone();
        after.patch(black_box(edits), &t);
        let a = attacks(&after, &t, false, avx);
        a[0] ^ a[1]
    });
    let pyramid_patch_query_ns = bench(children.len(), query_iters, |i| {
        let (_, edits, _, before) = &children[i];
        let mut after = before.clone();
        after.patch(black_box(edits), &t);
        let a = after.attacks(&t, avx);
        a[0] ^ a[1]
    });
    let union_ns = bench(children.len(), query_iters, |i| {
        let a = union_control(&children[i].2);
        a[0] ^ a[1]
    });
    emit(
        &mut output,
        json!({"kind":"transitions","count":children.len(),"iters":query_iters,"copy_patch_ns":patch_ns,"copy_patch_bands_query_ns":patch_query_ns,"copy_patch_pyramid_query_ns":pyramid_patch_query_ns,"union_child_query_ns":union_ns,"patch_input":"precomputed exact sparse edits; diff construction excluded"}),
    );
    if let Some(path) = arg("--out") {
        std::fs::write(
            path,
            output
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .unwrap();
    }
}
