//! Persistent typed-event experiment. No orthodox attacks inside the mechanisms.
use chen::core::{KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, PIECES};
use chen::{Color, Piece, Position};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{hint::black_box, sync::OnceLock, time::Instant};
struct AllocCounter;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
#[global_allocator]
static ALLOCATOR: AllocCounter = AllocCounter;
unsafe impl GlobalAlloc for AllocCounter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
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
type Board = [u8; 64];
fn code(c: Color, p: Piece) -> u8 {
    1 + p as u8 + 6 * c as u8
}
fn color(v: u8) -> usize {
    (v as usize - 1) / 6
}
fn kind(v: u8) -> usize {
    (v as usize - 1) % 6
}
fn slider(v: u8, diag: bool) -> bool {
    v != 0 && (kind(v) == 4 || kind(v) == if diag { 2 } else { 3 })
}
fn board(p: &Position) -> Board {
    let mut b = [0; 64];
    for c in [Color::White, Color::Black] {
        for k in PIECES {
            let mut bits = p.pieces(c, k);
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                b[s] = code(c, k);
            }
        }
    }
    b
}
#[derive(Clone, Copy, Default)]
struct Edit {
    s: u8,
    old: u8,
    new: u8,
}
#[derive(Clone, Copy, Default)]
struct Packet {
    edits: [Edit; 4],
    len: usize,
}
impl Packet {
    fn between(a: &Board, b: &Board) -> Self {
        let mut p = Self::default();
        for s in 0..64 {
            if a[s] != b[s] {
                assert!(p.len < 4);
                p.edits[p.len] = Edit {
                    s: s as u8,
                    old: a[s],
                    new: b[s],
                };
                p.len += 1;
            }
        }
        p
    }
    fn inverse(mut self) -> Self {
        for e in &mut self.edits[..self.len] {
            std::mem::swap(&mut e.old, &mut e.new);
        }
        self
    }
    fn apply(&self, b: &mut Board) {
        for e in &self.edits[..self.len] {
            debug_assert_eq!(b[e.s as usize], e.old);
            b[e.s as usize] = e.new;
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Line {
    s: [u8; 8],
    len: usize,
    diag: bool,
}
struct Geo {
    lines: [Line; 46],
    membership: [[(u8, u8); 4]; 64],
    lut: [[u8; 256]; 8],
    expand: [[u64; 256]; 46],
}
fn geo() -> &'static Geo {
    static GEO: OnceLock<Geo> = OnceLock::new();
    GEO.get_or_init(|| {
        let mut g = Geo {
            lines: [Line::default(); 46],
            membership: [[(0, 0); 4]; 64],
            lut: [[0; 256]; 8],
            expand: [[0; 256]; 46],
        };
        let mut n = 0;
        let mut mem = [0; 64];
        for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
            for s in 0..64 {
                let x = s % 8;
                let y = s / 8;
                let (px, py) = (x as i8 - dx, y as i8 - dy);
                if (0..8).contains(&px) && (0..8).contains(&py) {
                    continue;
                }
                let mut l = Line {
                    diag: dx != 0 && dy != 0,
                    ..Line::default()
                };
                let (mut xx, mut yy) = (x as i8, y as i8);
                while (0..8).contains(&xx) && (0..8).contains(&yy) {
                    let q = (yy * 8 + xx) as usize;
                    l.s[l.len] = q as u8;
                    g.membership[q][mem[q]] = (n as u8, l.len as u8);
                    mem[q] += 1;
                    l.len += 1;
                    xx += dx;
                    yy += dy;
                }
                g.lines[n] = l;
                n += 1;
            }
        }
        assert_eq!(n, 46);
        for from in 0..8 {
            for occ in 0..256 {
                let mut out = 0u8;
                for dir in [-1, 1] {
                    let mut to = from as i8 + dir;
                    while (0..8).contains(&to) {
                        out |= 1 << to;
                        if occ & (1 << to) != 0 {
                            break;
                        }
                        to += dir;
                    }
                }
                g.lut[from][occ] = out;
            }
        }
        for l in 0..46 {
            for mask in 0..256 {
                for slot in 0..g.lines[l].len {
                    if mask & (1 << slot) != 0 {
                        g.expand[l][mask] |= 1u64 << g.lines[l].s[slot];
                    }
                }
            }
        }
        g
    })
}
fn line_occ(b: &Board, l: usize) -> u8 {
    let l = &geo().lines[l];
    let mut occ = 0;
    for i in 0..l.len {
        occ |= ((b[l.s[i] as usize] != 0) as u8) << i;
    }
    occ
}
fn source(b: &Board, occ: &[u8; 46], s: usize) -> u64 {
    let v = b[s];
    if v == 0 {
        return 0;
    }
    match kind(v) {
        0 => PAWN_ATTACKS[color(v)][s],
        1 => KNIGHT_ATTACKS[s],
        5 => KING_ATTACKS[s],
        _ => {
            let mut a = 0;
            for &(l, slot) in &geo().membership[s] {
                let l = l as usize;
                if slider(v, geo().lines[l].diag) {
                    a |= geo().expand[l][geo().lut[slot as usize][occ[l] as usize] as usize];
                }
            }
            a
        }
    }
}
#[derive(Clone)]
struct Counts {
    counts: [[u8; 64]; 2],
    unions: [u64; 2],
}
impl Counts {
    fn new() -> Self {
        Self {
            counts: [[0; 64]; 2],
            unions: [0; 2],
        }
    }
    fn change(&mut self, c: usize, old: u64, new: u64) {
        let mut remove = old & !new;
        while remove != 0 {
            let s = remove.trailing_zeros() as usize;
            remove &= remove - 1;
            self.counts[c][s] -= 1;
            if self.counts[c][s] == 0 {
                self.unions[c] &= !(1 << s);
            }
        }
        let mut add = new & !old;
        while add != 0 {
            let s = add.trailing_zeros() as usize;
            add &= add - 1;
            self.counts[c][s] += 1;
            self.unions[c] |= 1 << s;
        }
    }
}
#[derive(Default, Clone, Copy)]
struct Stats {
    updates: u64,
    edits: u64,
    occ_events: u64,
    dirty: u64,
    recomputed: u64,
    changed: u64,
    targets: u64,
    aggregate_visited: u64,
    aggregate_written: u64,
    allocations: u64,
}
impl Stats {
    fn add(&mut self, o: Self) {
        self.updates += o.updates;
        self.edits += o.edits;
        self.occ_events += o.occ_events;
        self.dirty += o.dirty;
        self.recomputed += o.recomputed;
        self.changed += o.changed;
        self.targets += o.targets;
        self.aggregate_visited += o.aggregate_visited;
        self.aggregate_written += o.aggregate_written;
    }
}
trait Aggregation: Clone {
    fn metrics(&self) -> (u64, u64) {
        (0, 0)
    }
    fn fresh() -> Self;
    fn change(&mut self, c: usize, old: u64, new: u64);
    fn unions(&self) -> [u64; 2];
}
impl Aggregation for Counts {
    fn fresh() -> Self {
        Self::new()
    }
    fn change(&mut self, c: usize, old: u64, new: u64) {
        Counts::change(self, c, old, new)
    }
    fn unions(&self) -> [u64; 2] {
        self.unions
    }
}
// Heap leaves are source-square contributions, internal summaries are attack unions.
#[derive(Clone)]
struct Tree {
    nodes: [[u64; 128]; 2],
    source: usize,
    visited: u64,
    written: u64,
}
impl Aggregation for Tree {
    fn metrics(&self) -> (u64, u64) {
        (self.visited, self.written)
    }
    fn fresh() -> Self {
        Self {
            nodes: [[0; 128]; 2],
            source: 0,
            visited: 0,
            written: 0,
        }
    }
    fn change(&mut self, c: usize, _old: u64, new: u64) {
        let mut i = 64 + self.source;
        self.nodes[c][i] = new;
        self.written += 1;
        while i > 1 {
            self.visited += 1;
            i /= 2;
            let new = self.nodes[c][2 * i] | self.nodes[c][2 * i + 1];
            if self.nodes[c][i] == new {
                break;
            }
            self.nodes[c][i] = new;
            self.written += 1;
        }
    }
    fn unions(&self) -> [u64; 2] {
        [self.nodes[0][1], self.nodes[1][1]]
    }
}
/// Dense SIMD-friendly reduction ablation: sparse leaf edits, defer OR aggregation.
#[derive(Clone)]
struct Flat {
    leaves: [[u64; 64]; 2],
    source: usize,
}
impl Aggregation for Flat {
    fn fresh() -> Self {
        Self {
            leaves: [[0; 64]; 2],
            source: 0,
        }
    }
    fn change(&mut self, c: usize, _old: u64, new: u64) {
        self.leaves[c][self.source] = new;
    }
    fn unions(&self) -> [u64; 2] {
        [dense_union(&self.leaves[0]), dense_union(&self.leaves[1])]
    }
}
#[inline(never)]
fn dense_union(leaves: &[u64; 64]) -> u64 {
    leaves.iter().fold(0, |a, b| a | b)
}
trait SourceAggregation: Aggregation {
    fn set_source(&mut self, _s: usize) {}
}
impl SourceAggregation for Counts {}
impl SourceAggregation for Flat {
    fn set_source(&mut self, s: usize) {
        self.source = s;
    }
}
impl SourceAggregation for Tree {
    fn set_source(&mut self, s: usize) {
        self.source = s;
    }
}
#[derive(Clone)]
struct Persistent<A: SourceAggregation> {
    b: Board,
    occ: [u8; 46],
    contrib: [u64; 64],
    sliders: [u8; 46],
    counts: A,
}
type SourceCache = Persistent<Counts>;
type TreeCache = Persistent<Tree>;
type FlatCache = Persistent<Flat>;
impl<A: SourceAggregation> Persistent<A> {
    fn new(b: Board) -> Self {
        let mut s = Self {
            b,
            occ: [0; 46],
            contrib: [0; 64],
            sliders: [0; 46],
            counts: A::fresh(),
        };
        for l in 0..46 {
            s.occ[l] = line_occ(&b, l);
            for slot in 0..geo().lines[l].len {
                if slider(b[geo().lines[l].s[slot] as usize], geo().lines[l].diag) {
                    s.sliders[l] |= 1 << slot;
                }
            }
        }
        for q in 0..64 {
            s.contrib[q] = source(&b, &s.occ, q);
            if b[q] != 0 {
                s.counts.set_source(q);
                s.counts.change(color(b[q]), 0, s.contrib[q]);
            }
        }
        s
    }
    fn update(&mut self, p: &Packet, fine: bool) -> Stats {
        self.update_mode(p, if fine { 1 } else { 0 })
    }
    fn update_indexed(&mut self, p: &Packet) -> Stats {
        self.update_mode(p, 2)
    }
    fn update_mode(&mut self, p: &Packet, mode: usize) -> Stats {
        let metrics = self.counts.metrics();
        let mut st = Stats {
            updates: 1,
            edits: p.len as u64,
            ..Stats::default()
        };
        let mut dirty = if mode == 2 {
            discover_indexed(&self.occ, &self.sliders, p)
        } else {
            discover(&self.b, p, mode == 1)
        };
        let mut dirty_lines = 0u64;
        for e in &p.edits[..p.len] {
            for &(l, slot) in &geo().membership[e.s as usize] {
                if slider(e.old, geo().lines[l as usize].diag)
                    != slider(e.new, geo().lines[l as usize].diag)
                {
                    self.sliders[l as usize] ^= 1 << slot;
                }
            }
            if (e.old == 0) != (e.new == 0) {
                st.occ_events += 1;
                for &(l, slot) in &geo().membership[e.s as usize] {
                    dirty_lines |= 1 << l;
                    self.occ[l as usize] ^= 1 << slot;
                }
            }
        }
        p.apply(&mut self.b);
        if mode != 2 {
            dirty |= discover(&self.b, p, mode == 1);
        }
        st.dirty = dirty_lines.count_ones() as u64;
        st.recomputed = dirty.count_ones() as u64;
        while dirty != 0 {
            let s = dirty.trailing_zeros() as usize;
            dirty &= dirty - 1;
            let old_symbol = p.edits[..p.len]
                .iter()
                .find(|e| e.s as usize == s)
                .map_or(self.b[s], |e| e.old);
            let new_symbol = self.b[s];
            self.counts.set_source(s);
            let old = self.contrib[s];
            let new = source(&self.b, &self.occ, s);
            if old != new || old_symbol != new_symbol {
                st.changed += 1;
                st.targets += (old ^ new).count_ones() as u64;
                if old_symbol != 0 && new_symbol != 0 && color(old_symbol) == color(new_symbol) {
                    self.counts.change(color(old_symbol), old, new);
                } else {
                    if old_symbol != 0 {
                        self.counts.change(color(old_symbol), old, 0);
                    }
                    if new_symbol != 0 {
                        self.counts.change(color(new_symbol), 0, new);
                    }
                }
                self.contrib[s] = new;
            }
        }
        let after = self.counts.metrics();
        st.aggregate_visited = after.0 - metrics.0;
        st.aggregate_written = after.1 - metrics.1;
        st
    }
}
fn discover_indexed(occ: &[u8; 46], sliders: &[u8; 46], p: &Packet) -> u64 {
    let g = geo();
    let mut dirty = 0;
    for e in &p.edits[..p.len] {
        dirty |= 1 << e.s;
        if (e.old == 0) == (e.new == 0) {
            continue;
        }
        for &(l, slot) in &g.membership[e.s as usize] {
            let l = l as usize;
            let visible = g.lut[slot as usize][occ[l] as usize];
            dirty |= g.expand[l][(visible & sliders[l]) as usize];
        }
    }
    dirty
}
fn discover(b: &Board, p: &Packet, fine: bool) -> u64 {
    let mut dirty = 0u64;
    for e in &p.edits[..p.len] {
        dirty |= 1 << e.s;
        if (e.old == 0) == (e.new == 0) {
            continue;
        }
        if fine {
            let (x, y) = ((e.s % 8) as i8, (e.s / 8) as i8);
            for (d, &(dx, dy)) in DIRS.iter().enumerate() {
                let (mut xx, mut yy) = (x + dx, y + dy);
                while (0..8).contains(&xx) && (0..8).contains(&yy) {
                    let s = (yy * 8 + xx) as usize;
                    let v = b[s];
                    if v != 0 {
                        if slider(v, d >= 4) {
                            dirty |= 1 << s;
                        }
                        break;
                    }
                    xx += dx;
                    yy += dy;
                }
            }
        } else {
            for &(l, _) in &geo().membership[e.s as usize] {
                let l = &geo().lines[l as usize];
                for &s in &l.s[..l.len] {
                    if slider(b[s as usize], l.diag) {
                        dirty |= 1 << s;
                    }
                }
            }
        }
    }
    dirty
}
#[derive(Clone)]
struct LineCache {
    b: Board,
    occ: [u8; 46],
    channels: [[u64; 2]; 46],
    locals: [u64; 64],
    counts: Counts,
}
fn channel(b: &Board, occ: u8, id: usize) -> [u64; 2] {
    let g = geo();
    let l = &g.lines[id];
    let mut masks = [0u8; 2];
    for i in 0..l.len {
        let v = b[l.s[i] as usize];
        if slider(v, l.diag) {
            masks[color(v)] |= g.lut[i][occ as usize];
        }
    }
    [
        g.expand[id][masks[0] as usize],
        g.expand[id][masks[1] as usize],
    ]
}
impl LineCache {
    fn new(b: Board) -> Self {
        let mut s = Self {
            b,
            occ: [0; 46],
            channels: [[0; 2]; 46],
            locals: [0; 64],
            counts: Counts::new(),
        };
        for l in 0..46 {
            s.occ[l] = line_occ(&b, l);
            s.channels[l] = channel(&b, s.occ[l], l);
            for c in 0..2 {
                s.counts.change(c, 0, s.channels[l][c]);
            }
        }
        for q in 0..64 {
            if b[q] != 0 && matches!(kind(b[q]), 0 | 1 | 5) {
                s.locals[q] = source(&b, &s.occ, q);
                s.counts.change(color(b[q]), 0, s.locals[q]);
            }
        }
        s
    }
    fn update(&mut self, p: &Packet) -> Stats {
        let mut st = Stats {
            updates: 1,
            edits: p.len as u64,
            ..Stats::default()
        };
        let mut dirty = 0u64;
        for e in &p.edits[..p.len] {
            let occ_change = (e.old == 0) != (e.new == 0);
            if occ_change {
                st.occ_events += 1;
            }
            for &(l, slot) in &geo().membership[e.s as usize] {
                let line = &geo().lines[l as usize];
                if occ_change || slider(e.old, line.diag) || slider(e.new, line.diag) {
                    dirty |= 1 << l;
                }
                if occ_change {
                    self.occ[l as usize] ^= 1 << slot;
                }
            }
            let s = e.s as usize;
            if self.locals[s] != 0 {
                self.counts.change(color(e.old), self.locals[s], 0);
                self.locals[s] = 0;
            }
            self.b[s] = e.new;
            if e.new != 0 && matches!(kind(e.new), 0 | 1 | 5) {
                self.locals[s] = source(&self.b, &self.occ, s);
                self.counts.change(color(e.new), 0, self.locals[s]);
            }
        }
        st.dirty = dirty.count_ones() as u64;
        st.recomputed = st.dirty;
        while dirty != 0 {
            let l = dirty.trailing_zeros() as usize;
            dirty &= dirty - 1;
            let new = channel(&self.b, self.occ[l], l);
            for (c, &new_value) in new.iter().enumerate() {
                let old = self.channels[l][c];
                if old != new_value {
                    st.changed += 1;
                    st.targets += (old ^ new_value).count_ones() as u64;
                    self.counts.change(c, old, new_value);
                }
            }
            self.channels[l] = new;
        }
        st
    }
}
/// Demand-only channels: walk only rays incident on requested target, plus leaper LUTs.
fn attacked(b: &Board, by: usize, t: usize) -> bool {
    let mut knights = KNIGHT_ATTACKS[t];
    let mut kings = KING_ATTACKS[t];
    let mut pawns = PAWN_ATTACKS[1 - by][t];
    for (bits, k) in [(&mut knights, 1), (&mut kings, 5), (&mut pawns, 0)] {
        while *bits != 0 {
            let s = bits.trailing_zeros() as usize;
            *bits &= *bits - 1;
            let v = b[s];
            if v != 0 && color(v) == by && kind(v) == k {
                return true;
            }
        }
    }
    let (x, y) = ((t % 8) as i8, (t / 8) as i8);
    for (d, &(dx, dy)) in DIRS.iter().enumerate() {
        let (mut xx, mut yy) = (x + dx, y + dy);
        while (0..8).contains(&xx) && (0..8).contains(&yy) {
            let v = b[(yy * 8 + xx) as usize];
            if v != 0 {
                if color(v) == by && slider(v, d >= 4) {
                    return true;
                }
                break;
            }
            xx += dx;
            yy += dy;
        }
    }
    false
}
/// Predicate cache retains only a proof support. A true predicate needs one witness;
/// a false predicate needs every inspected first-blocker/leaper exclusion.
#[derive(Clone, Copy, Default)]
struct Predicate {
    target: usize,
    value: bool,
    deps: u64,
    valid: bool,
}
fn prove_attack(b: &Board, by: usize, t: usize) -> (bool, u64) {
    let mut deps = 0;
    for (mut bits, k) in [
        (KNIGHT_ATTACKS[t], 1),
        (KING_ATTACKS[t], 5),
        (PAWN_ATTACKS[1 - by][t], 0),
    ] {
        deps |= bits;
        while bits != 0 {
            let s = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let v = b[s];
            if v != 0 && color(v) == by && kind(v) == k {
                return (true, 1 << s);
            }
        }
    }
    let (x, y) = ((t % 8) as i8, (t / 8) as i8);
    for (d, &(dx, dy)) in DIRS.iter().enumerate() {
        let (mut xx, mut yy) = (x + dx, y + dy);
        let mut ray = 0;
        while (0..8).contains(&xx) && (0..8).contains(&yy) {
            let s = (yy * 8 + xx) as usize;
            ray |= 1 << s;
            let v = b[s];
            if v != 0 {
                if color(v) == by && slider(v, d >= 4) {
                    return (true, ray);
                }
                break;
            }
            xx += dx;
            yy += dy;
        }
        deps |= ray;
    }
    (false, deps)
}
struct LazySafety {
    b: Board,
    pred: [Predicate; 2],
}
impl LazySafety {
    fn new(b: Board) -> Self {
        Self {
            b,
            pred: [Predicate::default(); 2],
        }
    }
    fn apply(&mut self, p: &Packet) {
        let mut changed = 0;
        for e in &p.edits[..p.len] {
            changed |= 1 << e.s;
        }
        p.apply(&mut self.b);
        for pred in &mut self.pred {
            if pred.deps & changed != 0 {
                pred.valid = false;
            }
        }
    }
    fn query(&mut self, by: usize, target: usize) -> (bool, bool) {
        let p = &mut self.pred[by];
        if p.valid && p.target == target {
            return (p.value, false);
        }
        let (value, deps) = prove_attack(&self.b, by, target);
        *p = Predicate {
            target,
            value,
            deps,
            valid: true,
        };
        (value, true)
    }
}
fn oracle(p: &Position) -> [u64; 2] {
    let mut a = [0; 2];
    for c in [Color::White, Color::Black] {
        for s in 0..64 {
            if p.attackers_to(s, c) != 0 {
                a[c.index()] |= 1 << s;
            }
        }
    }
    a
}
fn scalar_union(p: &Position) -> [u64; 2] {
    let mut a = [0; 2];
    let occ = p.occupied();
    for c in [Color::White, Color::Black] {
        for k in PIECES {
            let mut bits = p.pieces(c, k);
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                a[c.index()] |= match k {
                    Piece::Pawn => PAWN_ATTACKS[c.index()][s],
                    Piece::Knight => KNIGHT_ATTACKS[s],
                    Piece::King => KING_ATTACKS[s],
                    Piece::Bishop => chen::core::bishop_attacks(s as u8, occ),
                    Piece::Rook => chen::core::rook_attacks(s as u8, occ),
                    Piece::Queen => {
                        chen::core::bishop_attacks(s as u8, occ)
                            | chen::core::rook_attacks(s as u8, occ)
                    }
                };
            }
        }
    }
    a
}
#[derive(Clone)]
struct Case {
    id: String,
    family: String,
    split: String,
    fen: String,
    p: Position,
    children: Vec<(Packet, Position)>,
}
fn cases(path: Option<&str>) -> Vec<Case> {
    let mut roots = Vec::new();
    if let Some(path) = path {
        for l in std::fs::read_to_string(path).unwrap().lines() {
            if l.starts_with('#') || l.trim().is_empty() {
                continue;
            }
            let f: Vec<_> = l.split(if l.contains('\t') { '\t' } else { '|' }).collect();
            if f.len() < 4 || f[0] == "id" {
                continue;
            }
            roots.push((
                f[0].to_string(),
                f[1].to_string(),
                f[2].to_string(),
                Position::from_fen(f[3]).unwrap(),
            ));
        }
    } else {
        for (i, fen) in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/P7/8/3pP3/8/8/7p/4K3 w - d6 0 1",
            "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",
        ]
        .iter()
        .enumerate()
        {
            roots.push((
                format!("fixture{i}"),
                "dev".into(),
                "special".into(),
                Position::from_fen(fen).unwrap(),
            ));
        }
        let mut p = Position::startpos();
        let mut seed = 0x12345678u64;
        for i in 0..768 {
            let moves = p.legal_moves();
            if moves.is_empty() || i % 96 == 95 {
                p = Position::startpos();
                continue;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            p.make_move(moves[seed as usize % moves.len()]);
            if i % 3 == 0 {
                roots.push((format!("walk{i}"), "dev".into(), "trajectory".into(), p));
            }
        }
    }
    roots
        .into_iter()
        .map(|(id, split, family, p)| {
            let b = board(&p);
            let children = p
                .legal_moves()
                .into_iter()
                .map(|m| {
                    let mut child = p;
                    child.make_move(m);
                    (Packet::between(&b, &board(&child)), child)
                })
                .collect();
            Case {
                id,
                split,
                family,
                fen: p.to_fen(),
                p,
                children,
            }
        })
        .collect()
}
fn validate(cases: &[Case]) -> u64 {
    let mut transitions = 0;
    for case in cases {
        let b = board(&case.p);
        let expected = oracle(&case.p);
        let mut coarse = SourceCache::new(b);
        let mut fine = coarse.clone();
        let mut indexed = coarse.clone();
        let mut itree = TreeCache::new(b);
        let mut flat = FlatCache::new(b);
        let mut tree = TreeCache::new(b);
        let mut lines = LineCache::new(b);
        assert_eq!(coarse.counts.unions(), expected);
        assert_eq!(lines.counts.unions(), expected);
        for (packet, p) in &case.children {
            let expected = oracle(p);
            coarse.update(packet, false);
            fine.update(packet, true);
            indexed.update_indexed(packet);
            itree.update_indexed(packet);
            flat.update_indexed(packet);
            tree.update(packet, true);
            lines.update(packet);
            assert_eq!(flat.counts.unions(), expected);
            assert_eq!(indexed.counts.unions(), expected, "{} indexed", case.id);
            assert_eq!(itree.counts.unions(), expected);
            assert_eq!(coarse.counts.unions(), expected, "{} coarse", case.id);
            assert_eq!(fine.counts.unions(), expected, "{} fine", case.id);
            assert_eq!(tree.counts.unions(), expected, "{} tree", case.id);
            assert_eq!(lines.counts.unions(), expected, "{} lines", case.id);
            assert_eq!(scalar_union(p), expected);
            for (c, &expected_mask) in expected.iter().enumerate() {
                for t in 0..64 {
                    assert_eq!(
                        attacked(&fine.b, c, t),
                        expected_mask & (1 << t) != 0,
                        "{} demand c={c} t={t}",
                        case.id
                    );
                }
            }
            let inverse = packet.inverse();
            indexed.update_indexed(&inverse);
            itree.update_indexed(&inverse);
            flat.update_indexed(&inverse);
            coarse.update(&inverse, false);
            fine.update(&inverse, true);
            tree.update(&inverse, true);
            lines.update(&inverse);
            assert_eq!(coarse.counts.unions(), oracle(&case.p));
            assert_eq!(fine.counts.unions(), coarse.counts.unions());
            assert_eq!(lines.counts.unions(), coarse.counts.unions());
            transitions += 1;
        }
        for decision in chen::codec::CodecPosition::from_position(&case.p)
            .relation()
            .decisions()
        {
            let mut p = case.p;
            p.make_move(decision.mv);
            let packet = Packet::between(&b, &board(&p));
            for e in decision.patch.edits() {
                assert_eq!(
                    board(&p)[e.square as usize],
                    e.piece.map_or(0, |(c, k)| code(c, k))
                );
            }
            fine.update(&packet, true);
            tree.update(&packet, true);
            let expected = oracle(&p);
            assert_eq!(
                fine.counts.unions(),
                expected,
                "{} pseudo {}",
                case.id,
                decision.mv
            );
            assert_eq!(tree.counts.unions(), expected);
            for (c, &expected_mask) in expected.iter().enumerate() {
                for t in 0..64 {
                    assert_eq!(attacked(&fine.b, c, t), expected_mask & (1 << t) != 0);
                }
            }
            fine.update(&packet.inverse(), true);
            tree.update(&packet.inverse(), true);
            transitions += 1;
        }
    }
    // True trajectories cover mutations that become the next ingress, without restoring snapshots.
    let mut p = Position::startpos();
    let mut fine = SourceCache::new(board(&p));
    let mut coarse = fine.clone();
    let mut tree = TreeCache::new(board(&p));
    let mut flat = FlatCache::new(board(&p));
    let mut lazy = LazySafety::new(board(&p));
    let mut lines = LineCache::new(board(&p));
    let mut seed = 7654321u64;
    for i in 0..4096 {
        let moves = p.legal_moves();
        if moves.is_empty() || i % 128 == 127 {
            p = Position::startpos();
            fine = SourceCache::new(board(&p));
            coarse = fine.clone();
            tree = TreeCache::new(board(&p));
            flat = FlatCache::new(board(&p));
            lazy = LazySafety::new(board(&p));
            lines = LineCache::new(board(&p));
            continue;
        }
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let before = board(&p);
        p.make_move(moves[seed as usize % moves.len()]);
        let packet = Packet::between(&before, &board(&p));
        fine.update_indexed(&packet);
        coarse.update(&packet, false);
        tree.update_indexed(&packet);
        flat.update_indexed(&packet);
        lazy.apply(&packet);
        lines.update(&packet);
        let expected = oracle(&p);
        for (by, t) in safety_targets(&p) {
            assert_eq!(lazy.query(by, t).0, expected[by] & (1 << t) != 0);
        }
        assert_eq!(tree.counts.unions(), expected);
        assert_eq!(flat.counts.unions(), expected);
        assert_eq!(fine.counts.unions(), expected);
        assert_eq!(coarse.counts.unions(), expected);
        assert_eq!(lines.counts.unions(), expected);
        transitions += 1;
    }
    transitions
}
#[derive(Clone, Copy, Debug)]
enum Method {
    Scalar,
    Rebuild,
    Coarse,
    Fine,
    TreeCoarse,
    TreeFine,
    Indexed,
    TreeIndexed,
    FlatIndexed,
    Lines,
    Demand,
    Lazy,
    Orthodox,
    Ingress,
}
impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar_full",
            Self::Rebuild => "source_rebuild",
            Self::Coarse => "source_coarse",
            Self::Fine => "source_fine",
            Self::TreeCoarse => "tree_coarse",
            Self::TreeFine => "tree_fine",
            Self::Indexed => "source_indexed",
            Self::TreeIndexed => "tree_indexed",
            Self::FlatIndexed => "flat_indexed",
            Self::Lines => "line_delta",
            Self::Demand => "demand_king",
            Self::Lazy => "lazy_king",
            Self::Orthodox => "orthodox_king",
            Self::Ingress => "ingress_packet",
        }
    }
}
const METHODS: [Method; 14] = [
    Method::Scalar,
    Method::Rebuild,
    Method::Coarse,
    Method::Fine,
    Method::TreeCoarse,
    Method::TreeFine,
    Method::Indexed,
    Method::TreeIndexed,
    Method::FlatIndexed,
    Method::Lines,
    Method::Demand,
    Method::Lazy,
    Method::Orthodox,
    Method::Ingress,
];
fn safety_targets(p: &Position) -> [(usize, usize); 2] {
    [
        (0, p.king_square(Color::Black) as usize),
        (1, p.king_square(Color::White) as usize),
    ]
}
fn run_siblings(cases: &[&Case], method: Method, reps: usize) -> (f64, u64, Stats) {
    let roots: Vec<_> = cases
        .iter()
        .map(|c| SourceCache::new(board(&c.p)))
        .collect();
    let line_roots: Vec<_> = cases.iter().map(|c| LineCache::new(board(&c.p))).collect();
    let tree_roots: Vec<_> = cases.iter().map(|c| TreeCache::new(board(&c.p))).collect();
    let flat_roots: Vec<_> = cases.iter().map(|c| FlatCache::new(board(&c.p))).collect();
    let n: usize = cases.iter().map(|c| c.children.len()).sum();
    let allocations_before = ALLOCS.load(Ordering::Relaxed);
    let start = Instant::now();
    let mut sum = 0u64;
    let mut stats = Stats::default();
    for _ in 0..reps {
        for (i, c) in cases.iter().enumerate() {
            let mut cache = matches!(method, Method::Coarse | Method::Fine | Method::Indexed)
                .then(|| roots[i].clone());
            let mut lc = matches!(method, Method::Lines).then(|| line_roots[i].clone());
            let mut tree = matches!(
                method,
                Method::TreeCoarse | Method::TreeFine | Method::TreeIndexed
            )
            .then(|| tree_roots[i].clone());
            let mut flat = matches!(method, Method::FlatIndexed).then(|| flat_roots[i].clone());
            let mut b = roots[i].b;
            let mut lazy = LazySafety::new(b);
            for (packet, p) in &c.children {
                let mut consume = None;
                match method {
                    Method::Scalar => {
                        consume = Some(scalar_union(black_box(p)));
                    }
                    Method::Rebuild => {
                        packet.apply(&mut b);
                        let s = SourceCache::new(black_box(b));
                        consume = Some(s.counts.unions());
                        packet.inverse().apply(&mut b);
                    }
                    Method::Coarse | Method::Fine => {
                        stats.add(
                            cache
                                .as_mut()
                                .unwrap()
                                .update(black_box(packet), matches!(method, Method::Fine)),
                        );
                        consume = Some(cache.as_ref().unwrap().counts.unions());
                        cache
                            .as_mut()
                            .unwrap()
                            .update(&packet.inverse(), matches!(method, Method::Fine));
                    }
                    Method::TreeCoarse | Method::TreeFine => {
                        stats.add(
                            tree.as_mut()
                                .unwrap()
                                .update(black_box(packet), matches!(method, Method::TreeFine)),
                        );
                        consume = Some(tree.as_ref().unwrap().counts.unions());
                        tree.as_mut()
                            .unwrap()
                            .update(&packet.inverse(), matches!(method, Method::TreeFine));
                    }
                    Method::Indexed => {
                        stats.add(cache.as_mut().unwrap().update_indexed(black_box(packet)));
                        consume = Some(cache.as_ref().unwrap().counts.unions());
                        cache.as_mut().unwrap().update_indexed(&packet.inverse());
                    }
                    Method::TreeIndexed => {
                        stats.add(tree.as_mut().unwrap().update_indexed(black_box(packet)));
                        consume = Some(tree.as_ref().unwrap().counts.unions());
                        tree.as_mut().unwrap().update_indexed(&packet.inverse());
                    }
                    Method::FlatIndexed => {
                        stats.add(flat.as_mut().unwrap().update_indexed(black_box(packet)));
                        consume = Some(flat.as_ref().unwrap().counts.unions());
                        flat.as_mut().unwrap().update_indexed(&packet.inverse());
                    }
                    Method::Lines => {
                        stats.add(lc.as_mut().unwrap().update(black_box(packet)));
                        consume = Some(lc.as_ref().unwrap().counts.unions());
                        lc.as_mut().unwrap().update(&packet.inverse());
                    }
                    Method::Demand => {
                        packet.apply(&mut b);
                        for (by, t) in safety_targets(p) {
                            sum += attacked(black_box(&b), by, t) as u64;
                        }
                        packet.inverse().apply(&mut b);
                    }
                    Method::Lazy => {
                        lazy.apply(packet);
                        for (by, t) in safety_targets(p) {
                            let (value, recomputed) = lazy.query(by, t);
                            sum += value as u64;
                            stats.recomputed += recomputed as u64;
                            stats.updates += 1;
                        }
                        lazy.apply(&packet.inverse());
                    }
                    Method::Orthodox => {
                        for (by, t) in safety_targets(p) {
                            sum += (black_box(p).attackers_to(
                                t as u8,
                                if by == 0 { Color::White } else { Color::Black },
                            ) != 0) as u64;
                        }
                    }
                    Method::Ingress => {
                        let packet = Packet::between(&board(black_box(&c.p)), &board(black_box(p)));
                        sum += packet.len as u64;
                    }
                }
                if let Some(a) = consume {
                    let a = black_box(a);
                    sum = sum.wrapping_add(a[0] ^ a[1]);
                }
            }
        }
    }
    black_box(sum);
    stats.allocations = ALLOCS.load(Ordering::Relaxed) - allocations_before;
    (
        start.elapsed().as_nanos() as f64 / (n * reps) as f64,
        sum,
        stats,
    )
}
struct Episode {
    start: Position,
    steps: Vec<(Packet, Position)>,
}
fn episodes() -> Vec<Episode> {
    let mut result = Vec::new();
    let mut seed = 0x15acefaceu64;
    for _ in 0..16 {
        let start = Position::startpos();
        let mut p = start;
        let mut steps = Vec::new();
        for _ in 0..128 {
            let moves = p.legal_moves();
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let b = board(&p);
            p.make_move(moves[seed as usize % moves.len()]);
            steps.push((Packet::between(&b, &board(&p)), p));
        }
        result.push(Episode { start, steps });
    }
    result
}
fn read_episodes(path: &str) -> Vec<Episode> {
    let mut result: Vec<Episode> = Vec::new();
    let mut last_id = String::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<_> = line.split('|').collect();
        assert_eq!(f.len(), 6);
        let before = Position::from_fen(f[3]).unwrap();
        let after = Position::from_fen(f[5]).unwrap();
        if f[1] != last_id {
            result.push(Episode {
                start: before,
                steps: Vec::new(),
            });
        } else {
            assert_eq!(result.last().unwrap().steps.last().unwrap().1, before);
        }
        result
            .last_mut()
            .unwrap()
            .steps
            .push((Packet::between(&board(&before), &board(&after)), after));
        last_id = f[0].to_string();
    }
    result
}
fn validate_episodes(episodes: &[Episode]) -> usize {
    let mut n = 0;
    for episode in episodes {
        let b = board(&episode.start);
        let mut coarse = SourceCache::new(b);
        let mut fine = SourceCache::new(b);
        let mut indexed = SourceCache::new(b);
        let mut tree = TreeCache::new(b);
        let mut flat = FlatCache::new(b);
        let mut lines = LineCache::new(b);
        let mut lazy = LazySafety::new(b);
        for (packet, p) in &episode.steps {
            coarse.update(packet, false);
            fine.update(packet, true);
            indexed.update_indexed(packet);
            tree.update_indexed(packet);
            flat.update_indexed(packet);
            lines.update(packet);
            lazy.apply(packet);
            assert_eq!(indexed.b, board(p));
            let expected = oracle(p);
            for actual in [
                coarse.counts.unions(),
                fine.counts.unions(),
                indexed.counts.unions(),
                tree.counts.unions(),
                flat.counts.unions(),
                lines.counts.unions(),
                scalar_union(p),
            ] {
                assert_eq!(actual, expected);
            }
            for (by, t) in safety_targets(p) {
                assert_eq!(lazy.query(by, t).0, expected[by] & (1 << t) != 0);
                assert_eq!(attacked(&indexed.b, by, t), expected[by] & (1 << t) != 0);
            }
            n += 1;
        }
    }
    n
}
fn run_trajectory(episodes: &[Episode], method: Method, reps: usize) -> (f64, u64, Stats) {
    let n: usize = episodes.iter().map(|e| e.steps.len()).sum();
    let allocations_before = ALLOCS.load(Ordering::Relaxed);
    let start = Instant::now();
    let mut sum = 0u64;
    let mut stats = Stats::default();
    for _ in 0..reps {
        for e in episodes {
            let mut cache = matches!(method, Method::Coarse | Method::Fine | Method::Indexed)
                .then(|| SourceCache::new(board(&e.start)));
            let mut lc = matches!(method, Method::Lines).then(|| LineCache::new(board(&e.start)));
            let mut tree = matches!(
                method,
                Method::TreeCoarse | Method::TreeFine | Method::TreeIndexed
            )
            .then(|| TreeCache::new(board(&e.start)));
            let mut flat =
                matches!(method, Method::FlatIndexed).then(|| FlatCache::new(board(&e.start)));
            let mut b = board(&e.start);
            let mut previous = e.start;
            let mut lazy = LazySafety::new(b);
            for (packet, p) in &e.steps {
                let mut consume = None;
                match method {
                    Method::Scalar => consume = Some(scalar_union(black_box(p))),
                    Method::Rebuild => {
                        packet.apply(&mut b);
                        let s = SourceCache::new(black_box(b));
                        consume = Some(s.counts.unions());
                    }
                    Method::Coarse | Method::Fine => {
                        stats.add(
                            cache
                                .as_mut()
                                .unwrap()
                                .update(black_box(packet), matches!(method, Method::Fine)),
                        );
                        consume = Some(cache.as_ref().unwrap().counts.unions());
                    }
                    Method::TreeCoarse | Method::TreeFine => {
                        stats.add(
                            tree.as_mut()
                                .unwrap()
                                .update(black_box(packet), matches!(method, Method::TreeFine)),
                        );
                        consume = Some(tree.as_ref().unwrap().counts.unions());
                    }
                    Method::Indexed => {
                        stats.add(cache.as_mut().unwrap().update_indexed(black_box(packet)));
                        consume = Some(cache.as_ref().unwrap().counts.unions());
                    }
                    Method::TreeIndexed => {
                        stats.add(tree.as_mut().unwrap().update_indexed(black_box(packet)));
                        consume = Some(tree.as_ref().unwrap().counts.unions());
                    }
                    Method::FlatIndexed => {
                        stats.add(flat.as_mut().unwrap().update_indexed(black_box(packet)));
                        consume = Some(flat.as_ref().unwrap().counts.unions());
                    }
                    Method::Lines => {
                        stats.add(lc.as_mut().unwrap().update(black_box(packet)));
                        consume = Some(lc.as_ref().unwrap().counts.unions());
                    }
                    Method::Demand => {
                        packet.apply(&mut b);
                        for (by, t) in safety_targets(p) {
                            sum += attacked(black_box(&b), by, t) as u64;
                        }
                    }
                    Method::Lazy => {
                        lazy.apply(packet);
                        for (by, t) in safety_targets(p) {
                            let (value, recomputed) = lazy.query(by, t);
                            sum += value as u64;
                            stats.recomputed += recomputed as u64;
                            stats.updates += 1;
                        }
                    }
                    Method::Orthodox => {
                        for (by, t) in safety_targets(p) {
                            sum += (black_box(p).attackers_to(
                                t as u8,
                                if by == 0 { Color::White } else { Color::Black },
                            ) != 0) as u64;
                        }
                    }
                    Method::Ingress => {
                        let packet =
                            Packet::between(&board(black_box(&previous)), &board(black_box(p)));
                        sum += packet.len as u64;
                        previous = *p;
                    }
                }
                if let Some(a) = consume {
                    let a = black_box(a);
                    sum = sum.wrapping_add(a[0] ^ a[1]);
                }
            }
        }
    }
    black_box(sum);
    stats.allocations = ALLOCS.load(Ordering::Relaxed) - allocations_before;
    (
        start.elapsed().as_nanos() as f64 / (n * reps) as f64,
        sum,
        stats,
    )
}
fn fresh(cases: &[Case], reps: usize, method: usize) -> (f64, u64) {
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..reps {
        for c in cases {
            let a = match method {
                0 => scalar_union(black_box(&c.p)),
                1 => SourceCache::new(board(black_box(&c.p))).counts.unions(),
                2 => LineCache::new(board(black_box(&c.p))).counts.unions(),
                3 => TreeCache::new(board(black_box(&c.p))).counts.unions(),
                _ => unreachable!(),
            };
            sum = sum.wrapping_add(a[0] ^ a[1]);
        }
    }
    black_box(sum);
    (
        start.elapsed().as_nanos() as f64 / (cases.len() * reps) as f64,
        sum,
    )
}
fn record(
    out: &mut String,
    mode: &str,
    family: &str,
    method: Method,
    reps: usize,
    n: usize,
    result: (f64, u64, Stats),
) {
    let (ns, sum, s) = result;
    out.push_str(&format!("{{\"kind\":\"timing\",\"mode\":\"{mode}\",\"family\":\"{family}\",\"method\":\"{}\",\"reps\":{reps},\"cases\":{n},\"ns_per_child\":{ns:.3},\"checksum\":{sum},\"updates\":{},\"edits\":{},\"occupancy_events\":{},\"dirty_lines\":{},\"recomputed\":{},\"changed\":{},\"changed_targets\":{},\"aggregate_visited\":{},\"aggregate_written\":{},\"allocations\":{}}}\n",method.name(),s.updates,s.edits,s.occ_events,s.dirty,s.recomputed,s.changed,s.targets,s.aggregate_visited,s.aggregate_written,s.allocations));
}
fn answers(cases: &[Case]) -> String {
    let mut out = String::new();
    for c in cases {
        let b = board(&c.p);
        let mut source = SourceCache::new(b);
        let mut tree = TreeCache::new(b);
        let mut lines = LineCache::new(b);
        let a = source.counts.unions();
        assert_eq!(a, tree.counts.unions());
        assert_eq!(a, lines.counts.unions());
        out.push_str(&format!(
            "{{\"id\":\"{}\",\"fen\":\"{}\",\"white\":{},\"black\":{}}}\n",
            c.id, c.fen, a[0], a[1]
        ));
        for (i, (packet, p)) in c.children.iter().enumerate() {
            source.update_indexed(packet);
            tree.update_indexed(packet);
            lines.update(packet);
            let a = source.counts.unions();
            assert_eq!(a, tree.counts.unions());
            assert_eq!(a, lines.counts.unions());
            out.push_str(&format!(
                "{{\"id\":\"{}#{i}\",\"fen\":\"{}\",\"white\":{},\"black\":{}}}\n",
                c.id,
                p.to_fen(),
                a[0],
                a[1]
            ));
            source.update_indexed(&packet.inverse());
            tree.update_indexed(&packet.inverse());
            lines.update(&packet.inverse());
        }
    }
    out
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let val = |name: &str| {
        args.iter()
            .position(|s| s == name)
            .map(|i| args[i + 1].as_str())
    };
    let path = val("--corpus");
    let reps = val("--reps").map_or(30, |s| s.parse::<usize>().unwrap());
    let mut cases = cases(path);
    if let Some(split) = val("--split") {
        cases.retain(|c| c.split == split);
    }
    if args.iter().any(|s| s == "--answers-only") {
        let out = answers(&cases);
        if let Some(path) = val("--out") {
            std::fs::write(path, &out).unwrap();
        } else {
            print!("{out}");
        }
        return;
    }
    let n: usize = cases.iter().map(|c| c.children.len()).sum();
    let verify_only = args.iter().any(|s| s == "--verify-only");
    let bench_only = args.iter().any(|s| s == "--bench-only");
    let mut out = String::new();
    out.push_str(&format!("{{\"kind\":\"metadata\",\"roots\":{},\"siblings\":{n},\"source_state_bytes\":{},\"line_state_bytes\":{},\"tree_state_bytes\":{},\"flat_state_bytes\":{},\"lazy_state_bytes\":{},\"board_bytes\":64,\"geometry_bytes\":{},\"allocations_per_update\":0,\"scope\":\"exact attack unions and king attack predicates; mate and strength N/A\"}}\n",cases.len(),std::mem::size_of::<SourceCache>(),std::mem::size_of::<LineCache>(),std::mem::size_of::<TreeCache>(),std::mem::size_of::<FlatCache>(),std::mem::size_of::<LazySafety>(),std::mem::size_of::<Geo>()));
    if !bench_only {
        let verified = validate(&cases);
        if let Some(path) = val("--transitions") {
            let n = validate_episodes(&read_episodes(path));
            out.push_str(&format!("{{\"kind\":\"shared_trajectory_validation\",\"transitions\":{n},\"boolean_answers\":{},\"all_exact\":true}}\n",n*128));
        }
        out.push_str(&format!("{{\"kind\":\"validation\",\"transitions\":{verified},\"boolean_answers\":{},\"all_exact\":true}}\n",verified*128));
    }
    if !verify_only {
        let mut methods = METHODS.to_vec();
        if args.iter().any(|s| s == "--reverse-methods") {
            methods.reverse();
        }
        if let Some(method) = val("--method") {
            methods.retain(|m| m.name() == method);
        }
        let all: Vec<_> = cases.iter().collect();
        for &method in &methods {
            record(
                &mut out,
                "siblings_roundtrip",
                "all",
                method,
                reps,
                n,
                run_siblings(&all, method, reps),
            );
        }
        let mut families: Vec<_> = cases.iter().map(|c| c.family.as_str()).collect();
        families.sort();
        families.dedup();
        for family in families {
            let subset: Vec<_> = cases.iter().filter(|c| c.family == family).collect();
            let count = subset.iter().map(|c| c.children.len()).sum();
            if count == 0 {
                continue;
            }
            for &method in &methods {
                record(
                    &mut out,
                    "siblings_roundtrip",
                    family,
                    method,
                    reps,
                    count,
                    run_siblings(&subset, method, reps),
                );
            }
        }
        let es = val("--transitions").map_or_else(episodes, read_episodes);
        for &method in &methods {
            record(
                &mut out,
                "trajectory",
                "generated",
                method,
                reps,
                es.iter().map(|e| e.steps.len()).sum(),
                run_trajectory(&es, method, reps),
            );
        }
        for (i, name) in ["scalar_full", "source_counts", "line_counts", "source_tree"]
            .iter()
            .enumerate()
        {
            let (ns, sum) = fresh(&cases, reps, i);
            out.push_str(&format!("{{\"kind\":\"timing\",\"mode\":\"fresh_ingress\",\"method\":\"{name}\",\"ns_per_root\":{ns:.3},\"checksum\":{sum}}}\n"));
        }
    }
    if let Some(path) = val("--out") {
        std::fs::write(path, &out).unwrap();
    }
    print!("{out}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deleting_one_attacker_preserves_the_other() {
        let before = Position::from_fen("7k/8/8/8/8/R7/8/R6K w - - 0 1").unwrap();
        let after = Position::from_fen("7k/8/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let packet = Packet::between(&board(&before), &board(&after));
        let mut counts = SourceCache::new(board(&before));
        let mut tree = TreeCache::new(board(&before));
        let mut flat = FlatCache::new(board(&before));
        let mut lines = LineCache::new(board(&before));
        counts.update_indexed(&packet);
        tree.update_indexed(&packet);
        flat.update_indexed(&packet);
        lines.update(&packet);
        let expected = oracle(&after);
        assert_ne!(expected[0] & (1 << 8), 0, "a2 remains attacked by a1 rook");
        assert_eq!(counts.counts.unions(), expected);
        assert_eq!(tree.counts.unions(), expected);
        assert_eq!(flat.counts.unions(), expected);
        assert_eq!(lines.counts.unions(), expected);
    }

    #[test]
    fn before_only_subscription_handles_multiple_removed_blockers() {
        let before = Position::from_fen("7k/8/8/8/p7/8/P7/R6K w - - 0 1").unwrap();
        let after = Position::from_fen("7k/8/8/8/1p6/8/1P6/R6K w - - 0 1").unwrap();
        let packet = Packet::between(&board(&before), &board(&after));
        let mut cache = FlatCache::new(board(&before));
        cache.update_indexed(&packet);
        assert_eq!(cache.counts.unions(), oracle(&after));
        cache.update_indexed(&packet.inverse());
        assert_eq!(cache.counts.unions(), oracle(&before));
    }

    #[test]
    fn true_predicate_keeps_only_witness_and_invalidates_on_witness_edit() {
        let before = Position::from_fen("7k/8/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let mut lazy = LazySafety::new(board(&before));
        assert_eq!(lazy.query(0, 56), (true, true)); // rook a1 attacks a8
        assert_eq!(lazy.pred[0].deps, 0x0001_0101_0101_0101);
        assert_eq!(lazy.query(0, 56), (true, false));
        let after = Position::from_fen("7k/8/8/8/8/8/P7/R6K w - - 0 1").unwrap();
        lazy.apply(&Packet::between(&board(&before), &board(&after)));
        assert_eq!(lazy.query(0, 56), (false, true));
    }

    #[test]
    fn special_move_and_trajectory_differential() {
        let fixture = Case {
            id: "castle-ep-promotion".into(),
            family: "test".into(),
            split: "dev".into(),
            fen: "4k3/P7/8/3pP3/8/8/7p/4K3 w - d6 0 1".into(),
            p: Position::from_fen("4k3/P7/8/3pP3/8/8/7p/4K3 w - d6 0 1").unwrap(),
            children: Vec::new(),
        };
        assert!(validate(&[fixture]) >= 4062);
    }
}
