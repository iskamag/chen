use chen::{Color, Position, codec::CodecPosition};

#[test]
fn whole_alternative_queries_match_individual_successors_across_word_boundaries() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "7k/8/8/Q1Q1Q1Q1/8/Q1Q1Q1Q1/8/K7 w - - 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        // Locally accepted, unreachable blocked board: no pseudo candidates.
        "NNNNNNNk/NNNNNNNN/NNNNNNNN/NNNNNNNN/NNNNNNNN/NNNNNNNN/NNNNNNNN/KNNNNNNN w - - 0 1",
        "4k3/P6r/8/8/8/8/7p/4K3 w - - 0 1",
    ] {
        let position = Position::from_fen(fen).unwrap();
        let relation = CodecPosition::from_position(&position).relation();
        let decisions = relation.decisions();
        let words = relation.legal_words();
        assert_eq!(words.len(), decisions.len().div_ceil(64));
        for color in [Color::White, Color::Black] {
            for square in 0..64 {
                let attacked: Vec<_> = relation.attacked_words(color, square).collect();
                assert_eq!(attacked.len(), words.len());
                for index in 0..words.len() * 64 {
                    let bit = 1 << (index % 64);
                    let legal = words[index / 64] & bit != 0;
                    let attack = attacked[index / 64] & bit != 0;
                    if let Some(decision) = decisions.get(index) {
                        let mut child = position;
                        child.make_move(decision.mv);
                        assert_eq!(legal, decision.legal, "{fen} / {index}");
                        assert_eq!(
                            attack,
                            child.attackers_to(square, color) != 0,
                            "{fen} / {index} / {square}"
                        );
                    } else {
                        assert!(!legal && !attack, "parent/transit/padding leaked");
                    }
                }
            }
        }
        let side = position.side_to_move();
        let checking = relation.attacked_words(side, position.king_square(side.opposite()));
        let exposed = relation.attacked_words(side.opposite(), 4);
        let selected: Vec<_> = words
            .iter()
            .zip(checking)
            .zip(exposed)
            .map(|((&legal, checking), exposed)| legal & checking & !exposed)
            .collect();
        for (index, decision) in decisions.iter().enumerate() {
            let mut child = position;
            child.make_move(decision.mv);
            let expected = decision.legal
                && child.is_in_check(side.opposite())
                && child.attackers_to(4, side.opposite()) == 0;
            assert_eq!(selected[index / 64] & (1 << (index % 64)) != 0, expected);
        }
    }
}
