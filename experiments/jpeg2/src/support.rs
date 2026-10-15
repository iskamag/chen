//! Identity-preserving support messages. These are geometric, pin-ray-filtered
//! obligations, not legal-capture, exchange, or tactical proofs. In particular,
//! king support and attacks from pinned enemies remain geometric observations.
use chen::codec::Patch;
use chen::core::{KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks};
use chen::{Color, Move, Piece, Position};
pub const DIM: usize = 64;
type Cell = Option<(Color, Piece)>;
const COST: [f32; 6] = [1., 3.2, 3.3, 5., 9., 0.];
const DIRS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
];
const METRICS: [&str; 12] = [
    "demand",
    "unsupported_demand",
    "sole_support_demand",
    "returned_pressure",
    "max_defender_load",
    "shared_defender_load",
    "shared_target_pairs",
    "pin_removed_edges",
    "support_edges",
    "attacked_value",
    "max_target_pressure",
    "pinned_value",
];
const CHANGES: [&str; 8] = [
    "lost_support_edges",
    "gained_support_edges",
    "abandoned_sole_value",
    "new_sole_value",
    "lost_defender_demand",
    "gained_defender_demand",
    "edited_sources_load_before",
    "edited_sources_load_after",
];
pub fn feature_names() -> Vec<String> {
    let mut out = Vec::with_capacity(DIM);
    for time in ["before", "after"] {
        for role in ["own", "enemy"] {
            for metric in METRICS {
                out.push(format!("support_{time}_{role}_{metric}"));
            }
        }
    }
    for role in ["own", "enemy"] {
        for metric in CHANGES {
            out.push(format!("support_change_{role}_{metric}"));
        }
    }
    out
}
/// Edges retain square identities: `defenders[target]` is a set of sources.
/// Source identity survives an action through the explicit move/patch mapping.
#[derive(Clone)]
pub struct SupportGraph {
    cells: [Cell; 64],
    pub defenders: [u64; 64],
    pub geometric_defenders: [u64; 64],
    pub attackers: [u64; 64],
    pub pin_rays: [u64; 64],
    pub demand: [f32; 64],
    pub load: [f32; 64],
    pub pressure: [f32; 64],
}
pub struct SupportRoot {
    pub graph: SupportGraph,
}
impl SupportRoot {
    pub fn new(p: &Position) -> Self {
        Self {
            graph: SupportGraph::build(std::array::from_fn(|s| p.piece_at(s as u8))),
        }
    }
    pub fn describe(&self, mv: Move, patch: &Patch, mover: Color) -> [f32; DIM] {
        let mut cells = self.graph.cells;
        for edit in patch.edits() {
            cells[edit.square as usize] = edit.piece;
        }
        let after = SupportGraph::build(cells);
        let mut out = [0.; DIM];
        for (time, graph) in [&self.graph, &after].iter().enumerate() {
            for (role, c) in [mover, mover.opposite()].iter().enumerate() {
                out[time * 24 + role * 12..time * 24 + role * 12 + 12]
                    .copy_from_slice(&graph.metrics(*c));
            }
        }
        // Track identities across captures, promotions, EP, and the castle rook.
        let mut mapping: [Option<usize>; 64] = std::array::from_fn(|s| {
            if self.graph.cells[s].is_some() && self.graph.cells[s] == cells[s] {
                Some(s)
            } else {
                None
            }
        });
        mapping[mv.from as usize] = Some(mv.to as usize);
        if self.graph.cells[mv.from as usize] == Some((mover, Piece::King))
            && mv.from.abs_diff(mv.to) == 2
        {
            let (rf, rt) = if mv.to > mv.from {
                (mv.from + 3, mv.from + 1)
            } else {
                (mv.from - 4, mv.from - 1)
            };
            mapping[rf as usize] = Some(rt as usize);
        }
        for (role, c) in [mover, mover.opposite()].iter().enumerate() {
            let f = &mut out[48 + role * 8..56 + role * 8];
            for t in 0..64 {
                let Some((owner, kind)) = self.graph.cells[t] else {
                    continue;
                };
                if owner != *c {
                    continue;
                }
                let Some(nt) = mapping[t] else { continue }; // Captured obligations disappear, not abandonment.
                let old = self.graph.defenders[t];
                let mut mapped = 0u64;
                for (s, ns) in mapping.iter().enumerate() {
                    if old & (1 << s) != 0
                        && let Some(ns) = ns
                    {
                        mapped |= 1 << ns;
                    }
                }
                let new = after.defenders[nt];
                // Count removed captured supporters as lost, even when no identity maps.
                let surviving = (mapped & new).count_ones();
                let lost = old.count_ones() - surviving;
                let gained = (new & !mapped).count_ones();
                f[0] += lost as f32 / 16.;
                f[1] += gained as f32 / 16.;
                if old.count_ones() == 1 && surviving == 0 {
                    f[2] += COST[kind.index()] / 16.;
                }
                if new.count_ones() == 1 && (new & !mapped) != 0 {
                    f[3] += COST[kind.index()] / 16.;
                }
                f[4] += self.graph.demand[t] * lost as f32 / 16.;
                f[5] += after.demand[nt] * gained as f32 / 16.;
            }
            for edit in patch.edits() {
                let s = edit.square as usize;
                if self.graph.cells[s].is_some_and(|(owner, _)| owner == *c) {
                    f[6] += self.graph.load[s] / 16.;
                }
                if after.cells[s].is_some_and(|(owner, _)| owner == *c) {
                    f[7] += after.load[s] / 16.;
                }
            }
        }
        out
    }
}
impl SupportGraph {
    fn build(cells: [Cell; 64]) -> Self {
        let mut g = Self {
            cells,
            defenders: [0; 64],
            geometric_defenders: [0; 64],
            attackers: [0; 64],
            pin_rays: [u64::MAX; 64],
            demand: [0.; 64],
            load: [0.; 64],
            pressure: [0.; 64],
        };
        let occupied = cells
            .iter()
            .enumerate()
            .fold(0, |b, (s, c)| b | if c.is_some() { 1 << s } else { 0 });
        for color in [Color::White, Color::Black] {
            let king = cells.iter().position(|c| *c == Some((color, Piece::King)));
            let Some(king) = king else { continue };
            for (di, (df, dr)) in DIRS.iter().enumerate() {
                let (mut f, mut r) = ((king % 8) as i8 + df, (king / 8) as i8 + dr);
                let mut first = None;
                let mut ray = 1u64 << king;
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    let s = (r * 8 + f) as usize;
                    ray |= 1 << s;
                    if let Some((c, k)) = cells[s] {
                        if first.is_none() && c == color {
                            first = Some(s);
                        } else {
                            if let Some(pin) = first
                                && c != color
                                && (k == Piece::Queen
                                    || (di < 4 && k == Piece::Rook)
                                    || (di >= 4 && k == Piece::Bishop))
                            {
                                g.pin_rays[pin] = ray;
                            }
                            break;
                        }
                    }
                    f += df;
                    r += dr;
                }
            }
        }
        for (s, cell) in cells.iter().enumerate() {
            let Some((c, k)) = *cell else { continue };
            let mut targets = match k {
                Piece::Pawn => PAWN_ATTACKS[c.index()][s],
                Piece::Knight => KNIGHT_ATTACKS[s],
                Piece::Bishop => bishop_attacks(s as u8, occupied),
                Piece::Rook => rook_attacks(s as u8, occupied),
                Piece::Queen => bishop_attacks(s as u8, occupied) | rook_attacks(s as u8, occupied),
                Piece::King => KING_ATTACKS[s],
            };
            targets &= occupied;
            while targets != 0 {
                let t = targets.trailing_zeros() as usize;
                targets &= targets - 1;
                if cells[t].unwrap().0 == c {
                    g.geometric_defenders[t] |= 1 << s;
                    if g.pin_rays[s] & (1 << t) != 0 {
                        g.defenders[t] |= 1 << s;
                    }
                } else {
                    g.attackers[t] |= 1 << s;
                }
            }
        }
        for (t, cell) in cells.iter().enumerate() {
            if let Some((_, kind)) = cell {
                // Quiet support has small demand; attacked pieces demand their value.
                // This prior is an observable for learning, never a tactical verdict.
                g.demand[t] = COST[kind.index()] * if g.attackers[t] != 0 { 1. } else { 0.125 };
            }
        }
        g.messages();
        g
    }
    fn messages(&mut self) {
        self.load = [0.; 64];
        self.pressure = [0.; 64];
        for t in 0..64 {
            let n = self.defenders[t].count_ones();
            if n == 0 {
                continue;
            }
            let mut sources = self.defenders[t];
            while sources != 0 {
                let s = sources.trailing_zeros() as usize;
                sources &= sources - 1;
                self.load[s] += self.demand[t] / n as f32;
            }
        }
        for t in 0..64 {
            let n = self.defenders[t].count_ones();
            if n == 0 {
                self.pressure[t] = self.demand[t];
                continue;
            }
            let mut sources = self.defenders[t];
            let mut competing = 0.;
            while sources != 0 {
                let s = sources.trailing_zeros() as usize;
                sources &= sources - 1;
                competing += (self.load[s] - self.demand[t] / n as f32).max(0.);
            }
            self.pressure[t] = self.demand[t] * competing / (n as f32 * (1. + competing));
        }
    }
    fn metrics(&self, color: Color) -> [f32; 12] {
        let mut f = [0f32; 12];
        let mut degrees = [0u32; 64];
        for t in 0..64 {
            let Some((c, k)) = self.cells[t] else {
                continue;
            };
            if c != color {
                continue;
            }
            let n = self.defenders[t].count_ones();
            f[0] += self.demand[t];
            if n == 0 {
                f[1] += self.demand[t];
            }
            if n == 1 {
                f[2] += self.demand[t];
            }
            f[3] += self.pressure[t];
            f[10] = f[10].max(self.pressure[t]);
            f[7] += (self.geometric_defenders[t] & !self.defenders[t]).count_ones() as f32;
            f[8] += n as f32;
            if self.attackers[t] != 0 {
                f[9] += COST[k.index()];
            }
            if self.pin_rays[t] != u64::MAX {
                f[11] += COST[k.index()];
            }
            for (s, d) in degrees.iter_mut().enumerate() {
                if self.defenders[t] & (1 << s) != 0 {
                    *d += 1;
                }
            }
        }
        for (s, d) in degrees.iter().enumerate() {
            if self.cells[s].is_none_or(|(c, _)| c != color) {
                continue;
            }
            f[4] = f[4].max(self.load[s]);
            if *d > 1 {
                f[5] += self.load[s];
                f[6] += (*d * (*d - 1) / 2) as f32;
            }
        }
        for v in &mut f {
            *v /= 16.;
        }
        f
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use chen::codec::CodecPosition;
    #[test]
    fn shared_identity_changes_messages_with_equal_target_counts() {
        let p = Position::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let mut shared = SupportRoot::new(&p).graph;
        shared.defenders = [0; 64];
        shared.demand = [0.; 64];
        shared.defenders[20] = 1 << 10;
        shared.defenders[21] = 1 << 10;
        shared.demand[20] = 5.;
        shared.demand[21] = 5.;
        shared.messages();
        let mut independent = shared.clone();
        independent.defenders[21] = 1 << 11;
        independent.messages();
        assert_eq!(
            shared.defenders[20].count_ones(),
            independent.defenders[20].count_ones()
        );
        assert_eq!(
            shared.defenders[21].count_ones(),
            independent.defenders[21].count_ones()
        );
        assert!(shared.pressure[20] > 4.);
        assert_eq!(independent.pressure[20], 0.);
    }
    #[test]
    fn pin_filters_off_ray_support_but_retains_along_ray() {
        // e2 rook pinned to e1 by e8 rook; d2 pawn only geometrically defended.
        let p = Position::from_fen("k3r3/8/8/8/8/8/3PR3/4K3 w - - 0 1").unwrap();
        let g = SupportRoot::new(&p).graph;
        assert_ne!(g.geometric_defenders[11] & (1 << 12), 0);
        assert_eq!(g.defenders[11] & (1 << 12), 0);
        assert_ne!(g.defenders[4] & (1 << 12), 0);
    }
    #[test]
    fn sparse_views_match_materialized_graphs_and_are_finite() {
        for fen in [
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
            "k3r3/8/8/8/8/8/3PR3/4K3 w - - 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let root = SupportRoot::new(&p);
            for d in CodecPosition::from_position(&p)
                .relation()
                .decisions()
                .iter()
                .filter(|d| d.legal)
            {
                let f = root.describe(d.mv, &d.patch, p.side_to_move());
                assert!(f.iter().all(|v| v.is_finite()));
                let mut child = p;
                child.make_move(d.mv);
                let oracle = SupportRoot::new(&child);
                for (role, c) in [p.side_to_move(), p.side_to_move().opposite()]
                    .iter()
                    .enumerate()
                {
                    assert_eq!(
                        &f[24 + role * 12..36 + role * 12],
                        &oracle.graph.metrics(*c)
                    );
                }
            }
        }
        assert_eq!(feature_names().len(), DIM);
    }
    #[test]
    fn quiet_move_abandons_identified_sole_support() {
        let p = Position::from_fen("7k/8/8/8/3P4/8/8/K2R4 w - - 0 1").unwrap();
        let root = SupportRoot::new(&p);
        let rel = CodecPosition::from_position(&p).relation();
        let d = rel
            .decisions()
            .iter()
            .find(|d| d.legal && d.mv.from == 3 && d.mv.to == 4)
            .unwrap();
        let f = root.describe(d.mv, &d.patch, Color::White);
        assert!(f[50] > 0.);
        assert!(f[52] > 0.);
    }
}
