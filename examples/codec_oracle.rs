//! Line protocol for independent rules oracles: FEN in, legal `move:check` pairs out.
use chen::{Position, codec::CodecPosition};
use std::io::{self, BufRead};

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.expect("read FEN");
        let position = Position::from_fen(&line).expect("valid FEN");
        let relation = CodecPosition::from_position(&position).relation();
        let mut moves: Vec<_> = relation
            .decisions()
            .iter()
            .filter(|decision| decision.legal)
            .map(|decision| format!("{}:{}", decision.mv, u8::from(decision.gives_check)))
            .collect();
        moves.sort_unstable();
        println!("{}", moves.join(" "));
    }
}
