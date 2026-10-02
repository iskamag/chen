//! Exact static baseline units, independent of search and teacher normalization.
//!
//! Compile with `rustc --edition=2024 -O examples/corpus_baselines.rs -o ...`.
//! These are Chen's actual source modules, not duplicated evaluation formulas.
//! Input: one FEN per line. Output: ordered JSON rows. Terminal adjudication is
//! deliberately separate: use the pinned corpus rules oracle and full history.
#[allow(dead_code)]
#[path = "../src/core.rs"]
mod core;
#[allow(dead_code)]
#[path = "../src/eval.rs"]
mod eval;

use core::Position;
use eval::{Evaluator, Material, Tapered};
use std::io::{self, BufRead, Read, Write};

fn evaluate(row: usize, fen: &str) -> Result<String, String> {
    let position = Position::from_fen(fen)?;
    Ok(format!(
        "{{\"row\":{row},\"hash\":{},\"native_material_cp\":{},\"native_tapered_cp\":{},\"kind\":\"raw_native_static_mover_CP_no_terminal_override\"}}",
        position.hash(),
        Material.evaluate(&position),
        Tapered.evaluate(&position)
    ))
}

fn run() -> Result<(), String> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut row = 0;
    loop {
        let mut line = String::new();
        let size = input
            .by_ref()
            .take(1025)
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if size == 0 {
            break;
        }
        if size > 1024 || !line.ends_with('\n') {
            return Err("FEN rows require a newline and at most 1024 bytes".into());
        }
        writeln!(output, "{}", evaluate(row, line.trim())?).map_err(|e| e.to_string())?;
        row += 1;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus baseline error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_static_units_and_mover_sign() {
        let white = Position::from_fen("7k/8/8/8/8/8/4P3/7K w - - 0 1").unwrap();
        let black = Position::from_fen("7k/8/8/8/8/8/4P3/7K b - - 0 1").unwrap();
        assert_eq!(Material.evaluate(&white), 100);
        assert_eq!(Material.evaluate(&black), -100);
        assert_eq!(Tapered.evaluate(&white), -Tapered.evaluate(&black));
        assert_ne!(Tapered.evaluate(&white), Material.evaluate(&white));
        assert_eq!(Material.evaluate(&Position::startpos()), 0);
        assert_eq!(Tapered.evaluate(&Position::startpos()), 0);
    }
}
