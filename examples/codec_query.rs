//! Intersect exact channel constraints without unpacking candidate positions.
use chen::{Position, codec::CodecPosition, core::parse_square};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let square = parse_square(args.get(1).expect("usage: codec_query SQUARE FEN")).unwrap();
    let position = Position::from_fen(&args[2..].join(" ")).expect("valid FEN");
    let side = position.side_to_move();
    let enemy = side.opposite();
    let relation = CodecPosition::from_position(&position).relation();
    let checking = relation.attacked_words(side, position.king_square(enemy));
    let exposed = relation.attacked_words(enemy, square);
    // Each AND/NOT reasons about up to 64 alternatives at once. The predicates
    // are exact: legal, giving check, and the requested square not attacked.
    // This does not imply the target holds a piece, or prove a win or best move.
    for (word, ((&legal, checking), exposed)) in relation
        .legal_words()
        .iter()
        .zip(checking)
        .zip(exposed)
        .enumerate()
    {
        let mut selected = legal & checking & !exposed;
        println!("word={word} selected={selected:016x}");
        while selected != 0 {
            let bit = selected.trailing_zeros() as usize;
            selected &= selected - 1;
            println!("{}", relation.decisions()[word * 64 + bit].mv);
        }
    }
}
