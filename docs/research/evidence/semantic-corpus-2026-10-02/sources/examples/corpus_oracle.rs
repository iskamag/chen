//! CPU corpus certification through Chen's actual rule kernel and resident arena.
//!
//! Input is one `FEN<TAB>comma-separated previous uint64 hashes in hex` per
//! line. A third tab field `children` requests exact legal successor snapshots.
//! Previous hashes exclude the current state. No teacher scores are emitted.
use chen::{
    Color, Move, Position,
    resident::{HostTree, ResidentTree, TreeSpan},
    rules::{RuleMove, RuleState},
};
use std::io::{self, BufRead, Read, Write};

const POLICY: &str =
    "chen_current_terminal_v1_mate_stalemate_then_halfmove100_insufficient_threefold_window";
const MAX_LINE: usize = 2 * 1024 * 1024;

fn action(position: &Position, mv: Move) -> u16 {
    let flip = if position.side_to_move() == Color::Black {
        56
    } else {
        0
    };
    let packet = RuleMove::try_from(mv).expect("kernel move is valid");
    ((u16::from(mv.from ^ flip) * 64 + u16::from(mv.to ^ flip)) * 5) + u16::from(packet.promotion)
}

fn snapshot(position: &Position, history: &[u64], arena: &mut HostTree) -> Result<String, String> {
    let state = RuleState::from(position);
    arena.reset(state, history).map_err(|e| e.to_string())?;
    // The actual arena classifies terminal nodes before material evaluation.
    // A classified terminal needs no evaluator call; evaluated_positions
    // therefore exposes terminal status without duplicating adjudication rules.
    let stats = arena
        .evaluate(TreeSpan { start: 0, count: 1 }, 0)
        .map_err(|e| e.to_string())?;
    let terminal = stats.evaluated_positions == 0;
    let mut legal = position.legal_moves();
    legal.sort_by_key(|&mv| action(position, mv));
    let actions = legal
        .iter()
        .map(|&mv| format!("{{\"action\":{},\"uci\":\"{}\"}}", action(position, mv), mv))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"boards\":{:?},\"fields\":[{},{},{},{},{}],\"hash\":{},\"terminal\":{},\"in_check\":{},\"insufficient\":{},\"legal\":[{}]}}",
        state.pieces,
        state.side,
        state.castling,
        state.ep_square,
        state.halfmove,
        state.fullmove,
        state.hash,
        terminal,
        position.is_in_check(position.side_to_move()),
        position.insufficient_material(),
        actions
    ))
}

fn parse_row(line: &str) -> Result<(Position, Vec<u64>, bool), String> {
    let fields: Vec<_> = line.split('\t').collect();
    if !(2..=3).contains(&fields.len()) || (fields.len() == 3 && fields[2] != "children") {
        return Err("expected FEN, previous hashes, and optional children tab field".into());
    }
    let p = Position::from_fen(fields[0])?;
    let history = if fields[1].is_empty() {
        Vec::new()
    } else {
        fields[1]
            .split(',')
            .map(|word| {
                if word.is_empty()
                    || word.len() > 16
                    || !word.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err("previous hashes must be unprefixed uint64 hexadecimal".into());
                }
                u64::from_str_radix(word, 16).map_err(|_| "invalid previous uint64 hash".into())
            })
            .collect::<Result<Vec<_>, String>>()?
    };
    if history.len() > u16::MAX as usize {
        return Err("history exceeds native arena bound".into());
    }
    Ok((p, history, fields.len() == 3))
}

fn run() -> Result<(), String> {
    let mut arena = HostTree::new(1)?;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    let mut line = String::new();
    let mut row = 0u64;
    loop {
        line.clear();
        // Bound allocation before read_line can absorb an arbitrary record.
        let bytes = input
            .by_ref()
            .take((MAX_LINE + 1) as u64)
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if bytes == 0 {
            break;
        }
        if bytes > MAX_LINE {
            return Err("input row exceeds 2 MiB".into());
        }
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        let (p, history, children) = parse_row(&line).map_err(|e| format!("row {row}: {e}"))?;
        let state = snapshot(&p, &history, &mut arena)?;
        let mut child_rows = Vec::new();
        if children {
            let mut legal = p.legal_moves();
            legal.sort_by_key(|&mv| action(&p, mv));
            let mut child_history = history.clone();
            child_history.push(p.hash());
            for mv in legal {
                let mut child = p;
                child.make_move(mv);
                let child_state = snapshot(&child, &child_history, &mut arena)?;
                child_rows.push(format!(
                    "{{\"action\":{},\"uci\":\"{}\",\"state\":{}}}",
                    action(&p, mv),
                    mv,
                    child_state
                ));
            }
        }
        writeln!(
            output,
            "{{\"row\":{},\"policy\":\"{}\",\"state\":{},\"children\":[{}]}}",
            row,
            POLICY,
            state,
            child_rows.join(",")
        )
        .map_err(|e| e.to_string())?;
        row = row.checked_add(1).ok_or("row index overflow")?;
    }
    output.flush().map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus oracle: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_is_native_and_history_window_is_bounded() {
        let p = Position::from_fen("7k/8/8/8/8/8/8/KR6 w - - 4 4294967295").unwrap();
        let mut arena = HostTree::new(1).unwrap();
        assert!(
            snapshot(&p, &[p.hash(), 0, p.hash()], &mut arena)
                .unwrap()
                .contains("\"terminal\":true")
        );
        assert!(
            snapshot(&p, &[p.hash(), p.hash(), 0, 0, 0, 0], &mut arena)
                .unwrap()
                .contains("\"terminal\":false")
        );
        assert!(
            snapshot(&p, &[], &mut arena)
                .unwrap()
                .contains("4294967295")
        );
    }
    #[test]
    fn grammar_and_orientation_are_explicit() {
        assert!(parse_row("bad").is_err());
        assert!(parse_row("7k/8/8/8/8/8/8/KR6 w - - 0 1\t0xFF").is_err());
        let p = Position::from_fen("7k/8/8/8/8/8/8/KR6 b - - 0 1").unwrap();
        let mv = p.parse_uci_move("h8g8").unwrap();
        assert_eq!(action(&p, mv), (7 * 64 + 6) * 5);
    }
}
