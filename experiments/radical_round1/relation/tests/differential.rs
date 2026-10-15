use chen::Position;
use chen_relation_r1::{Board, Mode, Verdict, classify, legal_actions, mate_threats, root_actions};
fn verify(p: &Position) {
    let (b, a) = root_actions(p);
    let mut actual: Vec<_> = a.iter().map(|x| x.mv.to_string()).collect();
    actual.sort();
    let mut expected: Vec<_> = p.legal_moves().iter().map(ToString::to_string).collect();
    expected.sort();
    assert_eq!(actual, expected, "{}", p.to_fen());
    for mode in [Mode::Stream, Mode::Full, Mode::Grouped] {
        let (v, _) = classify(&b, &a, mode, usize::MAX);
        for row in v {
            let mut child = *p;
            child.make_move(row.mv);
            let check = child.is_in_check(child.side_to_move());
            let has = child.has_legal_move();
            assert_eq!(row.check, check, "{} {}", p.to_fen(), row.mv);
            assert_eq!(
                row.verdict,
                if has {
                    Verdict::HasReply
                } else if check {
                    Verdict::Mate
                } else {
                    Verdict::Stalemate
                },
                "{} {} {mode:?}",
                p.to_fen(),
                row.mv
            );
            if let Some(w) = row.witness {
                assert!(child.legal_moves().contains(&w));
            }
        }
    }
}
#[test]
fn fixtures() {
    for fen in [
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    ] {
        verify(&Position::from_fen(fen).unwrap());
    }
}
#[test]
fn random_walks() {
    let mut seed = 918221u64;
    for _ in 0..20 {
        let mut p = Position::startpos();
        for _ in 0..120 {
            verify(&p);
            let moves = p.legal_moves();
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            p.make_move(moves[seed as usize % moves.len()]);
        }
    }
}
#[test]
fn budget_is_unknown() {
    let p = Position::startpos();
    let (b, a) = root_actions(&p);
    let (v, _) = classify(&b, &a, Mode::Stream, 0);
    assert!(v.iter().all(|x| x.verdict == Verdict::Unknown));
}
#[test]
fn nested_certificates() {
    let p = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 b - - 0 1").unwrap();
    let (b, a) = root_actions(&p);
    for (m, w, complete) in mate_threats(&b, &a, usize::MAX) {
        assert!(complete);
        let child = b.after(a.iter().find(|a| a.mv == m).unwrap());
        let replies = legal_actions(&child);
        let mut oracle = p;
        oracle.make_move(m);
        let mates: Vec<_> = oracle
            .legal_moves()
            .into_iter()
            .filter(|&r| {
                let mut t = oracle;
                t.make_move(r);
                t.is_in_check(t.side_to_move()) && !t.has_legal_move()
            })
            .collect();
        assert_eq!(w.is_some(), !mates.is_empty());
        assert_eq!(replies.len(), oracle.legal_moves().len());
    }
    let _ = Board::from_position(&p);
}
#[test]
fn complete_reply_relation_and_multiple_words() {
    for fen in [
        "7k/8/8/Q1Q1Q1Q1/8/Q1Q1Q1Q1/8/K7 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ] {
        let p = Position::from_fen(fen).unwrap();
        verify(&p);
        let (b, a) = root_actions(&p);
        chen_relation_r1::classify_observe(&b, &a, Mode::Full, usize::MAX, &mut |offset, rows| {
            for i in 0..(a.len() - offset).min(64) {
                let mut child = p;
                child.make_move(a[offset + i].mv);
                let mut expected: Vec<_> = child
                    .legal_moves()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                expected.sort();
                let mut actual: Vec<_> = rows
                    .iter()
                    .filter(|(_, mask)| mask & (1 << i) != 0)
                    .map(|(mv, _)| mv.to_string())
                    .collect();
                actual.sort();
                assert_eq!(actual, expected, "{fen} {}", a[offset + i].mv);
            }
        });
    }
}
