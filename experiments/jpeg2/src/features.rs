//! Relationship-specific action observables. No successor Position is made here.
//! Attack counts are geometric, including pinned attackers; learned factors must
//! distinguish these facts from profitable exchanges.
use chen::codec::{CodecPosition, Decision, MoveRelation, Patch};
use chen::core::{KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks};
use chen::{Color, Move, Piece, Position};
use serde_json::{Value, json};
pub const DIM: usize = 256;
pub const CHANNELS: [(&str, usize, usize); 7] = [
    ("action", 0, 32),
    ("material", 32, 48),
    ("pressure", 48, 112),
    ("destination_balance", 112, 144),
    ("slider_lines", 144, 192),
    ("knight_pawn", 192, 224),
    ("king_blocks", 224, 256),
];
const DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
];
const COST: [f32; 6] = [1.0, 3.2, 3.3, 5.0, 9.0, 20.0];
type Cell = Option<(Color, Piece)>;
#[derive(Clone)]
pub struct ActionFeatures {
    pub mv: Move,
    pub decision_index: usize,
    pub values: [f32; DIM],
}
pub struct RootFacts {
    p: Position,
    cells: [Cell; 64],
    pub relation: MoveRelation,
    sources: u64,
    root_attacks: [u64; 2],
    root_pressure: [f32; 12],
    root_counts: [f32; 12],
}
struct View<'a> {
    root: &'a RootFacts,
    patch: &'a Patch,
    occupancy: u64,
    sources: u64,
    kings: [u8; 2],
    index: usize,
}
impl RootFacts {
    pub fn new(p: &Position) -> Self {
        let cells = std::array::from_fn(|s| p.piece_at(s as u8));
        let relation = CodecPosition::from_position(p).relation();
        let mut root_attacks = [0; 2];
        for c in [Color::White, Color::Black] {
            for s in 0..64 {
                if relation.attacked_before(c, s) {
                    root_attacks[c.index()] |= 1 << s;
                }
            }
        }
        let mut root_pressure = [0.; 12];
        let mut root_counts = [0.; 12];
        for (s, cell) in cells.iter().enumerate() {
            if let Some((c, k)) = cell {
                let role = if *c == p.side_to_move() { 0 } else { 1 };
                let slot = role * 6 + k.index();
                root_counts[slot] += 1.;
                if root_attacks[c.opposite().index()] & (1 << s) != 0 {
                    root_pressure[slot] += 1.;
                }
            }
        }
        Self {
            p: *p,
            cells,
            relation,
            sources: p.occupied(),
            root_attacks,
            root_pressure,
            root_counts,
        }
    }
    pub fn actions(&self) -> Vec<ActionFeatures> {
        self.relation
            .decisions()
            .iter()
            .enumerate()
            .filter(|(_, d)| d.legal)
            .map(|(i, d)| ActionFeatures {
                mv: d.mv,
                decision_index: i,
                values: self.describe(i, d),
            })
            .collect()
    }
    fn role(&self, c: Color) -> usize {
        if c == self.p.side_to_move() { 0 } else { 1 }
    }
    fn describe(&self, index: usize, d: &Decision) -> [f32; DIM] {
        let v = View::new(self, index, d);
        let mut f = [0.; DIM];
        self.action(&v, d, &mut f[..32]);
        self.material(&v, &mut f[32..48]);
        self.pressure(&v, &mut f[48..112]);
        self.destination(&v, d, &mut f[112..144]);
        self.lines(&v, d, &mut f[144..192]);
        self.knight_pawn(&v, d, &mut f[192..224]);
        self.king_blocks(&v, d, &mut f[224..256]);
        debug_assert!(f.iter().all(|x| x.is_finite()));
        f
    }
    fn action(&self, v: &View<'_>, d: &Decision, f: &mut [f32]) {
        f[d.moved.index()] = 1.;
        if let Some(k) = d.captured {
            f[6 + k.index()] = 1.;
        }
        if let Some(k) = d.mv.promotion {
            f[12 + match k {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                Piece::Queen => 3,
                _ => unreachable!(),
            }] = 1.;
        }
        let from = self.orient(d.mv.from);
        let to = self.orient(d.mv.to);
        f[16] = (from % 8) as f32 / 7.;
        f[17] = (from / 8) as f32 / 7.;
        f[18] = (to % 8) as f32 / 7.;
        f[19] = (to / 8) as f32 / 7.;
        f[20] = ((to % 8) as i8 - (from % 8) as i8) as f32 / 7.;
        f[21] = ((to / 8) as i8 - (from / 8) as i8) as f32 / 7.;
        let own = self.p.side_to_move();
        let enemy = own.opposite();
        f[22] = distance(d.mv.from, self.p.king_square(enemy)) / 7.;
        f[23] = distance(d.mv.to, v.kings[enemy.index()]) / 7.;
        f[24] = distance(d.mv.from, self.p.king_square(own)) / 7.;
        f[25] = distance(d.mv.to, v.kings[own.index()]) / 7.;
        f[26] = (d.moved == Piece::King && d.mv.from.abs_diff(d.mv.to) == 2) as u8 as f32;
        f[27] = (d.moved == Piece::Pawn
            && d.captured.is_some()
            && self.cells[d.mv.to as usize].is_none()) as u8 as f32;
        f[28] = d.captured.is_some() as u8 as f32;
        f[29] = d.gives_check as u8 as f32;
        f[30] = d.mv.promotion.is_some() as u8 as f32;
        f[31] =
            (self.root_attacks[enemy.index()] & (1 << self.p.king_square(own)) != 0) as u8 as f32;
    }
    fn material(&self, v: &View<'_>, f: &mut [f32]) {
        f[..12].copy_from_slice(&self.root_counts);
        for edit in v.patch.edits() {
            if let Some((c, k)) = self.cells[edit.square as usize] {
                f[self.role(c) * 6 + k.index()] -= 1.;
            }
            if let Some((c, k)) = edit.piece {
                f[self.role(c) * 6 + k.index()] += 1.;
            }
        }
        for x in &mut f[..12] {
            *x /= 8.;
        }
        f[12] = f[1..5].iter().sum::<f32>() / 2.;
        f[13] = f[7..11].iter().sum::<f32>() / 2.;
        f[14] = v.occupancy.count_ones() as f32 / 32.;
        f[15] = (self.p.halfmove_clock() as f32 / 100.).min(2.);
    }
    fn pressure(&self, v: &View<'_>, f: &mut [f32]) {
        let own = self.p.side_to_move();
        let mut after = [0.; 12];
        for (s, c, k) in v.pieces() {
            let role = self.role(c);
            let a = v.attacked(c.opposite(), s);
            let defended = v.attacked(c, s);
            let state = (a as usize) * 2 + defended as usize;
            f[(role * 6 + k.index()) * 4 + state] += 1. / 8.;
            if a {
                after[role * 6 + k.index()] += 1.;
                if !defended
                    && matches!(
                        k,
                        Piece::Knight | Piece::Bishop | Piece::Rook | Piece::Queen
                    )
                {
                    f[60 + role] += 1. / 8.;
                }
            }
        }
        for i in 0..12 {
            f[48 + i] = (after[i] - self.root_pressure[i]) / 8.;
        }
        for (role, c) in [(0, own), (1, own.opposite())] {
            f[62 + role] = v
                .attackers(c.opposite(), v.kings[c.index()])
                .iter()
                .sum::<f32>()
                / 4.;
        }
    }
    fn destination(&self, v: &View<'_>, d: &Decision, f: &mut [f32]) {
        let own = self.p.side_to_move();
        let mut counts = [[0.; 6]; 4];
        for (role, c) in [(0, own), (1, own.opposite())] {
            counts[role] = v.attackers(c, d.mv.to);
            counts[2 + role] = self.root_attackers(c, d.mv.to);
            for k in 0..6 {
                f[role * 6 + k] = counts[role][k] / 4.;
                f[12 + role * 6 + k] = counts[2 + role][k] / 4.;
            }
        }
        for i in 0..4 {
            f[24 + i] = counts[i].iter().sum::<f32>() / 8.;
            f[28 + i] = counts[i]
                .iter()
                .enumerate()
                .filter(|(_, n)| **n > 0.)
                .map(|(k, _)| COST[k] / 20.)
                .fold(1.25, f32::min);
        }
    }
    fn lines(&self, v: &View<'_>, d: &Decision, f: &mut [f32]) {
        let sign = if self.p.side_to_move() == Color::White {
            1
        } else {
            -1
        };
        for (endpoint, square) in [d.mv.from, d.mv.to].iter().enumerate() {
            for (dir, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
                let (mut x, mut y) = ((*square % 8) as i8 + dx, (*square / 8) as i8 + dy * sign);
                let mut first = false;
                let mut distance = 1;
                let base = (endpoint * 8 + dir) * 3;
                while on_board(x, y) {
                    if let Some((c, k)) = v.cell((y * 8 + x) as u8) {
                        if !first {
                            f[base] = if self.role(c) == 0 { 1. } else { -1. }
                                * (k.index() + 1) as f32
                                / 6.;
                            f[base + 1] = distance as f32 / 7.;
                            first = true;
                        } else {
                            if k == Piece::King {
                                f[base + 2] = if self.role(c) == 0 { 1. } else { -1. };
                            }
                            break;
                        }
                    }
                    x += dx;
                    y += dy * sign;
                    distance += 1;
                }
            }
        }
    }
    fn knight_pawn(&self, v: &View<'_>, d: &Decision, f: &mut [f32]) {
        let own = self.p.side_to_move();
        let mut pawns = [Vec::new(), Vec::new()];
        let mut valuable = 0;
        for (s, c, k) in v.pieces() {
            let role = self.role(c);
            if KNIGHT_ATTACKS[d.mv.to as usize] & (1 << s) != 0 {
                f[role * 6 + k.index()] += 1. / 4.;
                if role == 1
                    && matches!(
                        k,
                        Piece::Knight | Piece::Bishop | Piece::Rook | Piece::Queen
                    )
                {
                    valuable += 1;
                }
            }
            if k == Piece::Pawn {
                pawns[role].push(s);
                let file = s % 8;
                let center = d.mv.to % 8;
                let delta = file as i8 - center as i8;
                if (-1..=1).contains(&delta) {
                    f[12 + role * 3 + (delta + 1) as usize] += 1. / 8.;
                }
                if PAWN_ATTACKS[c.index()][s as usize] & (1 << d.mv.to) != 0 {
                    f[18 + role] += 0.5;
                }
            }
        }
        for (role, c) in [(0, own), (1, own.opposite())] {
            for &s in &pawns[role] {
                let file = s % 8;
                let rank = relative_rank(c, s);
                let passed = !pawns[1 - role]
                    .iter()
                    .any(|&enemy| file.abs_diff(enemy % 8) <= 1 && relative_rank(c, enemy) > rank);
                if passed {
                    f[20 + role] += 1. / 8.;
                }
                if pawns[role]
                    .iter()
                    .any(|&friend| PAWN_ATTACKS[c.index()][friend as usize] & (1 << s) != 0)
                {
                    f[22 + role] += 1. / 8.;
                }
                if !pawns[role]
                    .iter()
                    .any(|&friend| file.abs_diff(friend % 8) == 1)
                {
                    f[24 + role] += 1. / 8.;
                }
                if pawns[role]
                    .iter()
                    .filter(|&&friend| friend % 8 == file)
                    .count()
                    > 1
                {
                    f[26 + role] += 1. / 8.;
                }
                if rank >= 4 {
                    f[28 + role] += 1. / 8.;
                }
            }
        }
        f[30] = (7 - relative_rank(own, d.mv.to)) as f32 / 7.;
        f[31] = (valuable >= 2) as u8 as f32;
    }
    fn king_blocks(&self, v: &View<'_>, d: &Decision, f: &mut [f32]) {
        let own = self.p.side_to_move();
        for (role, c) in [(0, own), (1, own.opposite())] {
            let king = v.kings[c.index()];
            let zone = KING_ATTACKS[king as usize];
            for target in bits(zone) {
                if v.attacked(c, target) {
                    f[role * 2] += 1. / 8.;
                }
                if v.attacked(c.opposite(), target) {
                    f[role * 2 + 1] += 1. / 8.;
                }
                if v.cell(target).is_none() && !v.attacked(c.opposite(), target) {
                    f[4 + role] += 1. / 8.;
                }
            }
            let attackers = v.attackers(c.opposite(), king);
            f[6 + role] = attackers[Piece::Knight.index()] / 4.;
            f[8 + role] = v.pins(c) as f32 / 8.;
            let file = king % 8;
            let rank = king / 8;
            let dy = if c == Color::White { 1 } else { -1 };
            for (dx, slot) in [(-1, 0), (0, 1), (1, 2)] {
                let (x, y) = (file as i8 + dx, rank as i8 + dy);
                if on_board(x, y) && v.cell((y * 8 + x) as u8) == Some((c, Piece::Pawn)) {
                    f[10 + role * 3 + slot] = 1.;
                }
            }
        }
        for (endpoint, s) in [d.mv.from, d.mv.to].iter().enumerate() {
            let x = (*s % 8) as i8;
            let y = (*s / 8) as i8;
            let vertical = if own == Color::White { 1 } else { -1 };
            for (q, (dx, dy)) in [(-1, -1), (0, -1), (-1, 0), (0, 0)].iter().enumerate() {
                for xx in 0..2 {
                    for yy in 0..2 {
                        let (a, b) = (x + dx + xx, y + (dy + yy) * vertical);
                        if on_board(a, b)
                            && let Some((c, _)) = v.cell((b * 8 + a) as u8)
                        {
                            f[16 + endpoint * 8 + q * 2 + self.role(c)] += 0.25;
                        }
                    }
                }
            }
        }
    }
    fn orient(&self, s: u8) -> u8 {
        if self.p.side_to_move() == Color::White {
            s
        } else {
            s ^ 56
        }
    }
    fn root_attackers(&self, c: Color, t: u8) -> [f32; 6] {
        let mut count = [0.; 6];
        for s in bits(self.sources) {
            if let Some((color, k)) = self.cells[s as usize]
                && color == c
                && source_attacks(s, color, k, self.sources) & (1 << t) != 0
            {
                count[k.index()] += 1.;
            }
        }
        count
    }
}
impl<'a> View<'a> {
    fn new(root: &'a RootFacts, index: usize, d: &'a Decision) -> Self {
        let mut occupancy = root.sources;
        let mut sources = root.sources;
        let mut kings = [
            root.p.king_square(Color::White),
            root.p.king_square(Color::Black),
        ];
        for e in d.patch.edits() {
            sources |= 1 << e.square;
            if e.piece.is_some() {
                occupancy |= 1 << e.square;
            } else {
                occupancy &= !(1 << e.square);
            }
            if let Some((c, Piece::King)) = e.piece {
                kings[c.index()] = e.square;
            }
        }
        Self {
            root,
            patch: &d.patch,
            occupancy,
            sources,
            kings,
            index,
        }
    }
    fn cell(&self, s: u8) -> Cell {
        self.patch
            .edits()
            .iter()
            .find(|e| e.square == s)
            .map_or(self.root.cells[s as usize], |e| e.piece)
    }
    fn pieces(&self) -> impl Iterator<Item = (u8, Color, Piece)> + '_ {
        bits(self.sources).filter_map(|s| self.cell(s).map(|(c, k)| (s, c, k)))
    }
    fn attacked(&self, by: Color, s: u8) -> bool {
        self.root.relation.attacked_after(self.index, by, s)
    }
    fn attackers(&self, by: Color, t: u8) -> [f32; 6] {
        let mut count = [0.; 6];
        for (s, c, k) in self.pieces() {
            if c == by && source_attacks(s, c, k, self.occupancy) & (1 << t) != 0 {
                count[k.index()] += 1.;
            }
        }
        count
    }
    fn pins(&self, c: Color) -> usize {
        let king = self.kings[c.index()];
        let mut pins = 0;
        for (dir, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let (mut x, mut y) = ((king % 8) as i8 + dx, (king / 8) as i8 + dy);
            let mut candidate = false;
            while on_board(x, y) {
                if let Some((color, k)) = self.cell((y * 8 + x) as u8) {
                    if !candidate {
                        if color != c || k == Piece::King {
                            break;
                        }
                        candidate = true;
                    } else {
                        if color != c
                            && (k == Piece::Queen
                                || k == if dir < 4 { Piece::Rook } else { Piece::Bishop })
                        {
                            pins += 1;
                        }
                        break;
                    }
                }
                x += dx;
                y += dy;
            }
        }
        pins
    }
}
fn source_attacks(s: u8, c: Color, k: Piece, occ: u64) -> u64 {
    match k {
        Piece::Pawn => PAWN_ATTACKS[c.index()][s as usize],
        Piece::Knight => KNIGHT_ATTACKS[s as usize],
        Piece::Bishop => bishop_attacks(s, occ),
        Piece::Rook => rook_attacks(s, occ),
        Piece::Queen => bishop_attacks(s, occ) | rook_attacks(s, occ),
        Piece::King => KING_ATTACKS[s as usize],
    }
}
fn on_board(x: i8, y: i8) -> bool {
    (0..8).contains(&x) && (0..8).contains(&y)
}
fn distance(a: u8, b: u8) -> f32 {
    a.abs_diff(b) as f32 * 0. + a.div_euclid(8).abs_diff(b / 8).max((a % 8).abs_diff(b % 8)) as f32
}
fn relative_rank(c: Color, s: u8) -> u8 {
    if c == Color::White { s / 8 } else { 7 - s / 8 }
}
fn bits(mut bits: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if bits == 0 {
            None
        } else {
            let s = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            Some(s)
        }
    })
}
pub fn schema() -> Value {
    json!({"version":1,"dim":DIM,"channels":CHANNELS.map(|(name,start,end)|json!({"name":name,"start":start,"end":end})),"orientation":"side-to-move rankflip+colorswap","semantics":"geometric attackers include pinned pieces; facts are not exchange proofs","scales":{"piece_counts":8,"destination_typed_attackers":4,"destination_total_attackers":8,"least_attacker_cost":20,"line_distance":7,"line_piece_signed_role":6,"king_zone_squares":8,"block_occupancy":4},"source_transition":"root plus sparse codec edit view; no successor Position created","feature_names":feature_names()})
}

pub fn feature_names() -> Vec<String> {
    let pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"];
    let roles = ["own", "enemy"];
    let mut names: Vec<String> = [
        "move_pawn",
        "move_knight",
        "move_bishop",
        "move_rook",
        "move_queen",
        "move_king",
        "capture_pawn",
        "capture_knight",
        "capture_bishop",
        "capture_rook",
        "capture_queen",
        "capture_king",
        "promote_knight",
        "promote_bishop",
        "promote_rook",
        "promote_queen",
        "from_file",
        "from_rank",
        "to_file",
        "to_rank",
        "delta_file",
        "delta_rank",
        "from_enemy_king_distance",
        "to_enemy_king_distance",
        "from_own_king_distance",
        "to_own_king_distance",
        "castling",
        "en_passant",
        "capture",
        "gives_check",
        "promotion",
        "root_in_check",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for role in roles {
        for piece in pieces {
            names.push(format!("material_{role}_{piece}"));
        }
    }
    names.extend(
        [
            "own_nonpawn_count",
            "enemy_nonpawn_count",
            "occupied_count",
            "root_halfmove_clock",
        ]
        .map(String::from),
    );
    for role in roles {
        for piece in pieces {
            for state in [
                "unattacked_undefended",
                "unattacked_defended",
                "attacked_undefended",
                "attacked_defended",
            ] {
                names.push(format!("pressure_{role}_{piece}_{state}"));
            }
        }
    }
    for role in roles {
        for piece in pieces {
            names.push(format!("threat_delta_{role}_{piece}"));
        }
    }
    names.extend(
        [
            "own_undefended_valuable_count",
            "enemy_undefended_valuable_count",
            "own_king_checker_count",
            "enemy_king_checker_count",
        ]
        .map(String::from),
    );
    for phase in ["after", "root"] {
        for role in roles {
            for piece in pieces {
                names.push(format!("destination_{phase}_{role}_{piece}_attacker_count"));
            }
        }
    }
    for phase in ["after", "root"] {
        for role in roles {
            names.push(format!("destination_{phase}_{role}_attacker_total"));
        }
    }
    for phase in ["after", "root"] {
        for role in roles {
            names.push(format!(
                "destination_{phase}_{role}_least_attacker_cost_or_absent"
            ));
        }
    }
    for endpoint in ["source", "destination"] {
        for direction in [
            "east",
            "west",
            "forward",
            "backward",
            "forward_east",
            "backward_west",
            "backward_east",
            "forward_west",
        ] {
            for summary in [
                "first_piece_signed_type",
                "first_piece_distance",
                "second_piece_king_signed_role",
            ] {
                names.push(format!("line_{endpoint}_{direction}_{summary}"));
            }
        }
    }
    for role in roles {
        for piece in pieces {
            names.push(format!("knight_landing_{role}_{piece}_targets"));
        }
    }
    for role in roles {
        for file in ["left", "center", "right"] {
            names.push(format!("pawn_window_{role}_{file}_count"));
        }
    }
    for feature in [
        "destination_pawn_attackers",
        "passed_pawns",
        "supported_pawns",
        "isolated_pawns",
        "doubled_pawns",
        "advanced_pawns",
    ] {
        for role in roles {
            names.push(format!("{feature}_{role}"));
        }
    }
    names.extend(
        [
            "destination_promotion_distance",
            "knight_landing_fork_two_valuable",
        ]
        .map(String::from),
    );
    for role in roles {
        for by in ["friendly", "hostile"] {
            names.push(format!("king_zone_{role}_controlled_by_{by}"));
        }
    }
    for feature in [
        "king_safe_empty_neighbors",
        "king_knight_checkers",
        "absolute_pins",
    ] {
        for role in roles {
            names.push(format!("{feature}_{role}"));
        }
    }
    for role in roles {
        for file in ["left", "center", "right"] {
            names.push(format!("king_shield_{role}_{file}_pawn"));
        }
    }
    for endpoint in ["source", "destination"] {
        for quadrant in ["back_left", "back_right", "front_left", "front_right"] {
            for role in roles {
                names.push(format!("block_{endpoint}_{quadrant}_{role}_occupancy"));
            }
        }
    }
    assert_eq!(names.len(), DIM);
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn named_schema_covers_every_feature() {
        assert_eq!(feature_names().len(), DIM);
    }
    #[test]
    fn sparse_view_attack_source_counts_match_independent_position_oracle() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/P7/8/3pP3/8/8/7p/4K3 w - d6 0 1",
            "4k3/8/8/8/3Pp3/8/7P/4K3 b - d3 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let root = RootFacts::new(&p);
            for (index, d) in root
                .relation
                .decisions()
                .iter()
                .enumerate()
                .filter(|(_, d)| d.legal)
            {
                let v = View::new(&root, index, d);
                let mut child = p;
                child.make_move(d.mv);
                assert_eq!(v.occupancy, child.occupied());
                for c in [Color::White, Color::Black] {
                    for t in 0..64 {
                        let expected = child.attackers_to(t, c);
                        let actual = v.attackers(c, t);
                        for k in chen::core::PIECES {
                            assert_eq!(
                                actual[k.index()],
                                (expected & child.pieces(c, k)).count_ones() as f32,
                                "{fen} {} target={t} {c:?} {k:?}",
                                d.mv
                            );
                        }
                        assert_eq!(v.attacked(c, t), expected != 0);
                    }
                }
                assert!(root.describe(index, d).iter().all(|x| x.is_finite()));
            }
        }
    }
}

/// Binds the semantic feature contract, independently of model grouping/width.
pub fn schema_fingerprint() -> String {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(&schema()).expect("schema serializes");
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub const SUPPORT_DIM: usize = DIM + crate::support::DIM;
pub fn schema_for_dimension(dim: usize) -> Option<Value> {
    if dim == DIM {
        return Some(schema());
    }
    if dim != SUPPORT_DIM {
        return None;
    }
    let mut v = schema();
    v["version"] = json!(2);
    v["dim"] = json!(SUPPORT_DIM);
    v["channels"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"support_obligations","start":DIM,"end":SUPPORT_DIM}));
    v["feature_names"].as_array_mut().unwrap().extend(
        crate::support::feature_names()
            .into_iter()
            .map(Value::String),
    );
    v["support_semantics"] = json!(
        "source-identity support incidence; absolute-pin-ray filtering; two-pass target demand to shared defender load to target pressure; geometric, not an exchange proof"
    );
    v["support_computation"] = json!(
        "root graph cached once; each action copies 64 symbols and reconstructs its counterfactual graph; no successor Position or game-tree reasoning"
    );
    Some(v)
}
pub fn schema_fingerprint_for(dim: usize) -> Option<String> {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(&schema_for_dimension(dim)?).ok()?;
    Some(
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}
impl RootFacts {
    pub fn feature_rows(
        &self,
        actions: &[ActionFeatures],
        dim: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if dim != DIM && dim != SUPPORT_DIM {
            return Err(format!("unsupported feature dimension {dim}"));
        }
        let support = (dim == SUPPORT_DIM).then(|| crate::support::SupportRoot::new(&self.p));
        Ok(actions
            .iter()
            .map(|action| {
                let mut row = Vec::with_capacity(dim);
                row.extend_from_slice(&action.values);
                if let Some(root) = &support {
                    let d = &self.relation.decisions()[action.decision_index];
                    row.extend_from_slice(&root.describe(
                        action.mv,
                        &d.patch,
                        self.p.side_to_move(),
                    ));
                }
                row
            })
            .collect())
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    #[test]
    fn frozen_contracts_have_distinct_stable_fingerprints() {
        assert_eq!(
            schema_fingerprint(),
            "890278fadee8d12099254c2e1edc428a84157977ddffe5a32335f9b690ed8d5f"
        );
        assert_eq!(
            schema_fingerprint_for(SUPPORT_DIM).unwrap(),
            "d70273878dad4b215e50d13bf24c3c045c8eaffb43a2a6e6f974244f419fd5d2"
        );
        assert!(schema_fingerprint_for(257).is_none());
    }
}
