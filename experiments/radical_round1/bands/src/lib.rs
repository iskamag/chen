//! Exact topology transforms. This crate is an experiment, not an engine policy.
use chen::{Color, Piece, Position};

const A: u64 = 0xfefefefefefefefe;
const H: u64 = 0x7f7f7f7f7f7f7f7f;
const COLORS: [Color; 2] = [Color::White, Color::Black];
const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[derive(Clone)]
pub struct Tables {
    pub micro: Vec<u128>,
    /// 4 cells, 4 source bits, incoming active bit -> attacks + outgoing bit.
    forward: Vec<i32>,
    backward: Vec<i32>,
    /// [operator (pawnW/pawnB/knight/king)][source row][source byte].
    band: Vec<u64>,
    spread: Vec<u64>,
    geometry: Vec<Vec<u8>>,
    square_lines: [[(u8, u8); 4]; 64],
    first_two: Vec<u32>,
    rays: [[u64; 8]; 64],
}

fn offsets(p: usize, color: usize) -> Vec<(i8, i8)> {
    match p {
        0 => vec![
            (-1, if color == 0 { 1 } else { -1 }),
            (1, if color == 0 { 1 } else { -1 }),
        ],
        1 => vec![
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ],
        5 => vec![
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
        _ => vec![],
    }
}

impl Tables {
    pub fn new() -> Self {
        let mut micro = vec![0u128; 13usize.pow(4)];
        for (code, out) in micro.iter_mut().enumerate() {
            let mut code = code;
            for cell in 0..4 {
                let piece = code % 13;
                code /= 13;
                if piece == 0 {
                    continue;
                }
                let color = (piece - 1) / 6;
                let p = (piece - 1) % 6;
                *out |= 1u128 << (88 + cell);
                if p == 3 || p == 4 {
                    *out |= 1u128 << (72 + color * 4 + cell);
                }
                if p == 2 || p == 4 {
                    *out |= 1u128 << (80 + color * 4 + cell);
                }
                for (df, dr) in offsets(p, color) {
                    let f = 2 + (cell % 2) as i8 + df;
                    let r = 2 + (cell / 2) as i8 + dr;
                    *out |= 1u128 << (color * 36 + (r * 6 + f) as usize);
                }
            }
        }
        let mut forward = vec![0i32; 512];
        let mut backward = vec![0i32; 512];
        for index in 0..512 {
            for (table, reverse) in [(&mut forward, false), (&mut backward, true)] {
                let mut active = index & 256 != 0;
                let mut attacks = 0;
                for k in 0..4 {
                    let bit = if reverse { 3 - k } else { k };
                    if active {
                        attacks |= 1 << bit;
                    }
                    active = (active && index & (1 << bit) == 0) || index & (1 << (4 + bit)) != 0;
                }
                table[index] = attacks | if active { 16 } else { 0 };
            }
        }
        let mut band = vec![0u64; 4 * 2 * 256];
        for op in 0..4 {
            for row in 0..2 {
                for mask in 0..256 {
                    let (p, c) = match op {
                        0 => (0, 0),
                        1 => (0, 1),
                        2 => (1, 0),
                        _ => (5, 0),
                    };
                    for file in 0..8 {
                        if mask & (1 << file) == 0 {
                            continue;
                        }
                        for (df, dr) in offsets(p, c) {
                            let f = file as i8 + df;
                            let r = 2 + row as i8 + dr;
                            if (0..8).contains(&f) && (0..6).contains(&r) {
                                band[(op * 2 + row) * 256 + mask] |= 1u64 << (r * 8 + f);
                            }
                        }
                    }
                }
            }
        }
        let mut geometry: Vec<Vec<u8>> = Vec::new();
        for rank in 0..8 {
            geometry.push((0..8).map(|file| rank * 8 + file).collect());
        }
        for file in 0..8 {
            geometry.push((0..8).map(|rank| rank * 8 + file).collect());
        }
        for d in -7i8..=7 {
            geometry.push(
                (0..8i8)
                    .filter_map(|r| {
                        let f = r + d;
                        (0..8).contains(&f).then_some((r * 8 + f) as u8)
                    })
                    .collect(),
            );
        }
        for d in 0..=14i8 {
            geometry.push(
                (0..8i8)
                    .filter_map(|r| {
                        let f = d - r;
                        (0..8).contains(&f).then_some((r * 8 + f) as u8)
                    })
                    .collect(),
            );
        }
        let mut square_lines = [[(0, 0); 4]; 64];
        let mut counts = [0; 64];
        let mut spread = vec![0u64; geometry.len() * 256];
        for (line, squares) in geometry.iter().enumerate() {
            for (i, &sq) in squares.iter().enumerate() {
                square_lines[sq as usize][counts[sq as usize]] = (line as u8, i as u8);
                counts[sq as usize] += 1;
            }
            for mask in 0..256 {
                for (i, &sq) in squares.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        spread[line * 256 + mask] |= 1u64 << sq;
                    }
                }
            }
        }
        let mut first_two = vec![0u32; 8 * 256];
        for king in 0..8 {
            for occ in 0..256 {
                let left: Vec<_> = (0..king).rev().filter(|b| occ & (1 << b) != 0).collect();
                let right: Vec<_> = (king + 1..8).filter(|b| occ & (1 << b) != 0).collect();
                let mut result = 0;
                for (j, v) in [left.first(), left.get(1), right.first(), right.get(1)]
                    .iter()
                    .enumerate()
                {
                    result |= v.map_or(8, |v| *v as u32) << (j * 4);
                }
                first_two[king * 256 + occ] = result;
            }
        }
        let mut rays = [[0u64; 8]; 64];
        for (sq, square_rays) in rays.iter_mut().enumerate() {
            for (d, (df, dr)) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ]
            .iter()
            .enumerate()
            {
                let mut f = (sq % 8) as i8 + df;
                let mut r = (sq / 8) as i8 + dr;
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    square_rays[d] |= 1u64 << (r * 8 + f);
                    f += df;
                    r += dr;
                }
            }
        }
        Self {
            micro,
            forward,
            backward,
            band,
            spread,
            geometry,
            square_lines,
            first_two,
            rays,
        }
    }

    pub fn bytes(&self) -> usize {
        self.micro.len() * 16
            + self.forward.len() * 4
            + self.backward.len() * 4
            + self.band.len() * 8
            + self.spread.len() * 8
            + self.first_two.len() * 4
            + std::mem::size_of_val(&self.rays)
    }
    pub fn line_count(&self) -> usize {
        self.geometry.len()
    }

    /// Exhaustive operator-equivalence measurements. These are equivalence
    /// classes of named answers, not equivalence classes of chess futures.
    pub fn operator_classes(&self) -> (usize, [usize; 8]) {
        use std::collections::HashSet;
        let mut whole_attacks = HashSet::new();
        for raw in 0..65536u32 {
            let mut x = raw;
            let mut code = 0;
            for bit in 0..8 {
                let class = x & 3;
                x >>= 2;
                if class != 0 {
                    code |= 1 << bit;
                }
                if class >= 2 {
                    code |= 1 << (8 + (class - 2) * 8 + bit);
                }
            }
            whole_attacks.insert(line_attacks(code, 0, self) | line_attacks(code, 1, self) << 8);
        }
        let mut targets = [0; 8];
        // A fixed white king and seven cells over empty, white blocker,
        // black blocker, white emitter, black emitter. Five source classes
        // preserve ownership needed for absolute-pin obligations.
        for (king, count) in targets.iter_mut().enumerate() {
            let mut classes = HashSet::new();
            for raw in 0..5u32.pow(7) {
                let mut x = raw;
                let mut occ = 1 << king;
                let mut own = 1 << king;
                let mut enemy_emit = 0;
                for bit in 0..8 {
                    if bit == king {
                        continue;
                    }
                    let class = x % 5;
                    x /= 5;
                    if class != 0 {
                        occ |= 1 << bit;
                    }
                    if class == 1 || class == 3 {
                        own |= 1 << bit;
                    }
                    if class == 4 {
                        enemy_emit |= 1 << bit;
                    }
                }
                let boundary = self.first_two[king * 256 + occ];
                let mut checker = 0u32;
                let mut pin = 0u32;
                let mut evasion = 255u32;
                for (a, b) in [(0, 4), (8, 12)] {
                    let first = (boundary >> a) & 15;
                    let second = (boundary >> b) & 15;
                    if first >= 8 {
                        continue;
                    }
                    if enemy_emit & (1 << first) != 0 {
                        checker |= 1 << first;
                        let lo = first.min(king as u32);
                        let hi = first.max(king as u32);
                        let ray = ((1u32 << (hi + 1)) - (1u32 << lo)) & !(1 << king);
                        evasion &= ray;
                    } else if own & (1 << first) != 0
                        && second < 8
                        && enemy_emit & (1 << second) != 0
                    {
                        pin |= 1 << first;
                    }
                }
                classes.insert(checker | pin << 8 | evasion << 16);
            }
            *count = classes.len();
        }
        (whole_attacks.len(), targets)
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

/// Exact redundant channels, rather than a minimum-size board serialization.
/// Lines contain occupancy and color-specific matching slider emitter bits.
#[derive(Clone)]
pub struct Encoded {
    pub lines: [u32; 48],
    pub micro: [u16; 16],
    pub planes: [[u64; 6]; 2],
    pub kings: [u8; 2],
    pub board: [u8; 64],
}

fn symbol(piece: Option<(Color, Piece)>) -> u8 {
    piece.map_or(0, |(c, p)| 1 + 6 * c as u8 + p as u8)
}

impl Encoded {
    pub fn new(position: &Position, t: &Tables) -> Self {
        let mut s = Self {
            lines: [0; 48],
            micro: [0; 16],
            planes: [[0; 6]; 2],
            kings: [
                position.king_square(Color::White),
                position.king_square(Color::Black),
            ],
            board: [0; 64],
        };
        for sq in 0..64 {
            s.set(sq, symbol(position.piece_at(sq)), t);
        }
        s
    }
    fn set(&mut self, sq: u8, new: u8, t: &Tables) {
        let old = self.board[sq as usize];
        if old == new {
            return;
        }
        let tile = (sq / 16) * 4 + (sq % 8) / 2;
        let cell = (sq / 8) % 2 * 2 + sq % 2;
        self.micro[tile as usize] = (self.micro[tile as usize] as i32
            + (new as i32 - old as i32) * 13i32.pow(cell as u32))
            as u16;
        for p in [old, new] {
            if p == 0 {
                continue;
            }
            let c = (p - 1) / 6;
            let pc = (p - 1) % 6;
            self.planes[c as usize][pc as usize] ^= 1u64 << sq;
            if pc == 5 && p == new {
                self.kings[c as usize] = sq;
            }
            for &(line, index) in &t.square_lines[sq as usize] {
                self.lines[line as usize] ^= 1u32 << index;
                if pc == 4 || if line < 16 { pc == 3 } else { pc == 2 } {
                    self.lines[line as usize] ^= 1u32 << (8 + 8 * c + index);
                }
            }
        }
        self.board[sq as usize] = new;
    }
    /// Sparse semantic edits are supplied by caller; no 64-square diff is hidden here.
    pub fn patch(&mut self, edits: &[(u8, u8)], t: &Tables) {
        for &(sq, p) in edits {
            self.set(sq, p, t);
        }
    }
}

pub fn edits(before: &Position, after: &Position) -> Vec<(u8, u8)> {
    (0..64)
        .filter_map(|sq| {
            let a = symbol(before.piece_at(sq));
            let b = symbol(after.piece_at(sq));
            (a != b).then_some((sq, b))
        })
        .collect()
}

fn segment(index: usize, occ: u32, source: u32, incoming: u32) -> usize {
    ((occ >> (4 * index)) & 15) as usize
        | (((source >> (4 * index)) & 15) as usize) << 4
        | ((incoming & 16) as usize) << 4
}

#[inline]
fn line_attacks(code: u32, color: usize, t: &Tables) -> usize {
    let occ = code & 255;
    let src = (code >> (8 + 8 * color)) & 255;
    let lo = t.forward[segment(0, occ, src, 0)] as u32;
    let hi = t.forward[segment(1, occ, src, lo)] as u32;
    let back_hi = t.backward[segment(1, occ, src, 0)] as u32;
    let back_lo = t.backward[segment(0, occ, src, back_hi)] as u32;
    ((lo | back_lo) & 15 | ((hi | back_hi) & 15) << 4) as usize
}

#[inline(never)]
pub fn slider_scalar(s: &Encoded, t: &Tables) -> [u64; 2] {
    let mut attacks = [0; 2];
    for (c, out) in attacks.iter_mut().enumerate() {
        for (line, &code) in s.lines[..t.geometry.len()].iter().enumerate() {
            *out |= t.spread[line * 256 + line_attacks(code, c, t)];
        }
    }
    attacks
}

/// Eight independent topology-line transducers, with vector table decoding.
/// Four gathers compose two four-cell blocks in both directions. This does not
/// vectorize alternatives and does not rely on compiler auto-vectorization.
///
/// # Safety
/// Caller must first establish AVX2 CPU support (see `avx_available`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn slider_avx2(s: &Encoded, t: &Tables) -> [u64; 2] {
    use std::arch::x86_64::*;
    unsafe {
        let mut result = [0; 2];
        let nibble = _mm256_set1_epi32(15);
        let bit4 = _mm256_set1_epi32(16);
        for (c, out) in result.iter_mut().enumerate() {
            for base in (0..48).step_by(8) {
                let code = _mm256_loadu_si256(s.lines.as_ptr().add(base).cast());
                let shift = _mm_cvtsi32_si128((8 + 8 * c) as i32);
                let src = _mm256_srl_epi32(code, shift);
                let lo_index = _mm256_or_si256(
                    _mm256_and_si256(code, nibble),
                    _mm256_slli_epi32::<4>(_mm256_and_si256(src, nibble)),
                );
                let hi_index = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi32::<4>(code), nibble),
                    _mm256_and_si256(src, _mm256_set1_epi32(240)),
                );
                let lo = _mm256_i32gather_epi32::<4>(t.forward.as_ptr(), lo_index);
                let hi = _mm256_i32gather_epi32::<4>(
                    t.forward.as_ptr(),
                    _mm256_or_si256(hi_index, _mm256_slli_epi32::<4>(_mm256_and_si256(lo, bit4))),
                );
                let bh = _mm256_i32gather_epi32::<4>(t.backward.as_ptr(), hi_index);
                let bl = _mm256_i32gather_epi32::<4>(
                    t.backward.as_ptr(),
                    _mm256_or_si256(lo_index, _mm256_slli_epi32::<4>(_mm256_and_si256(bh, bit4))),
                );
                let masks = _mm256_or_si256(
                    _mm256_and_si256(_mm256_or_si256(lo, bl), nibble),
                    _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_or_si256(hi, bh), nibble)),
                );
                let mut decoded = [0i32; 8];
                _mm256_storeu_si256(decoded.as_mut_ptr().cast(), masks);
                // The line lengths vary. Padding lines decode to zero and are
                // omitted instead of indexing beyond the expansion table.
                for (k, &mask) in decoded.iter().enumerate() {
                    if base + k < t.geometry.len() {
                        *out |= t.spread[(base + k) * 256 + mask as usize];
                    }
                }
            }
        }
        result
    }
}

pub fn avx_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn translate_stencil(stencil: u64, file: i32, rank: i32) -> u64 {
    let mut result = 0;
    for r in 0..6 {
        let board_rank = rank + r - 2;
        if !(0..8).contains(&board_rank) {
            continue;
        }
        let row = (stencil >> (6 * r)) & 63;
        let shifted = if file >= 2 {
            row << (file - 2)
        } else {
            row >> (2 - file)
        };
        result |= (shifted & 255) << (board_rank * 8);
    }
    result
}

#[inline(never)]
pub fn leapers_micro(s: &Encoded, t: &Tables) -> [u64; 2] {
    let mut result = [0; 2];
    for (tile, &code) in s.micro.iter().enumerate() {
        let value = t.micro[code as usize];
        for (c, out) in result.iter_mut().enumerate() {
            *out |= translate_stencil(
                ((value >> (36 * c)) & ((1u128 << 36) - 1)) as u64,
                ((tile % 4) * 2) as i32,
                ((tile / 4) * 2) as i32,
            );
        }
    }
    result
}

#[inline(never)]
pub fn leapers_band(s: &Encoded, t: &Tables) -> [u64; 2] {
    let mut result = [0; 2];
    for (c, out) in result.iter_mut().enumerate() {
        for pair in 0..4 {
            let mut local = 0;
            for row in 0..2 {
                let shift = pair * 16 + row * 8;
                for (p, op) in [(0, c), (1, 2), (5, 3)] {
                    let source = ((s.planes[c][p] >> shift) & 255) as usize;
                    local |= t.band[(op * 2 + row) * 256 + source];
                }
            }
            *out |= if pair == 0 {
                local >> 16
            } else {
                local << (pair * 16 - 16)
            };
        }
    }
    result
}

pub fn attacks(s: &Encoded, t: &Tables, micro: bool, avx: bool) -> [u64; 2] {
    let slider = if avx {
        assert!(avx_available());
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { slider_avx2(s, t) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!()
        }
    } else {
        slider_scalar(s, t)
    };
    let leaper = if micro {
        leapers_micro(s, t)
    } else {
        leapers_band(s, t)
    };
    [slider[0] | leaper[0], slider[1] | leaper[1]]
}

/// Four 4x4 macroblocks composed from the exhaustive 2x2 semantic dictionary.
/// Each macroblock retains only its exact projected leaper consequence; ray
/// obligations remain in the orthogonal/diagonal transfer channels. Incoming
/// updates touch changed microblocks and their macroblock ancestors only.
#[derive(Clone)]
pub struct Pyramid {
    pub state: Encoded,
    macros: [[u64; 2]; 4],
}

impl Pyramid {
    pub fn new(position: &Position, t: &Tables) -> Self {
        let mut result = Self {
            state: Encoded::new(position, t),
            macros: [[0; 2]; 4],
        };
        for block in 0..4 {
            result.refresh(block, t);
        }
        result
    }
    fn refresh(&mut self, block: usize, t: &Tables) {
        let mut stencil = [0u64; 2];
        for dy in 0..2 {
            for dx in 0..2 {
                let tile = (block / 2 * 2 + dy) * 4 + block % 2 * 2 + dx;
                let micro = t.micro[self.state.micro[tile] as usize];
                for (c, out) in stencil.iter_mut().enumerate() {
                    let local = (micro >> (36 * c)) as u64;
                    for row in 0..6 {
                        *out |= ((local >> (6 * row)) & 63) << ((row + 2 * dy) * 8 + 2 * dx);
                    }
                }
            }
        }
        for (c, local) in stencil.iter().enumerate() {
            let mut global = 0;
            let file = (block % 2 * 4) as i32 - 2;
            let rank = (block / 2 * 4) as i32 - 2;
            for row in 0..8 {
                let r = rank + row;
                if !(0..8).contains(&r) {
                    continue;
                }
                let bits = (local >> (row * 8)) & 255;
                let projected = if file >= 0 {
                    bits << file
                } else {
                    bits >> (-file)
                };
                global |= (projected & 255) << (r * 8);
            }
            self.macros[block][c] = global;
        }
    }
    /// Returns refreshed macroblock count, enabling real causal-width evidence.
    pub fn patch(&mut self, edits: &[(u8, u8)], t: &Tables) -> usize {
        let old_codes = self.state.micro;
        self.state.patch(edits, t);
        let mut dirty = 0u8;
        for (tile, (&old, &new)) in old_codes.iter().zip(&self.state.micro).enumerate() {
            // Only the leaper consequence is consumed by this pyramid.
            // Changes solely to ray emitters do not dirty a macroblock.
            if (t.micro[old as usize] ^ t.micro[new as usize]) & ((1u128 << 72) - 1) != 0 {
                dirty |= 1 << ((tile / 8) * 2 + (tile % 4) / 2);
            }
        }
        for block in 0..4 {
            if dirty & (1 << block) != 0 {
                self.refresh(block, t);
            }
        }
        dirty.count_ones() as usize
    }
    #[inline(never)]
    pub fn attacks(&self, t: &Tables, avx: bool) -> [u64; 2] {
        let mut out = if avx {
            #[cfg(target_arch = "x86_64")]
            {
                assert!(avx_available());
                unsafe { slider_avx2(&self.state, t) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!()
            }
        } else {
            slider_scalar(&self.state, t)
        };
        for block in &self.macros {
            out[0] |= block[0];
            out[1] |= block[1];
        }
        out
    }
}

fn fill<const SHIFT: u32, const UP: bool>(mut g: u64, mut p: u64) -> u64 {
    if UP {
        g |= p & (g << SHIFT);
        p &= p << SHIFT;
        g |= p & (g << (2 * SHIFT));
        p &= p << (2 * SHIFT);
        g |= p & (g << (4 * SHIFT));
        g << SHIFT
    } else {
        g |= p & (g >> SHIFT);
        p &= p >> SHIFT;
        g |= p & (g >> (2 * SHIFT));
        p &= p >> (2 * SHIFT);
        g |= p & (g >> (4 * SHIFT));
        g >> SHIFT
    }
}

/// Strong multi-source whole-board control, not a loop over 128 target queries.
#[inline(never)]
pub fn union_control(p: &Position) -> [u64; 2] {
    let mut out = [0; 2];
    let empty = !p.occupied();
    for c in COLORS {
        let i = c.index();
        let rook = p.pieces(c, Piece::Rook) | p.pieces(c, Piece::Queen);
        let bishop = p.pieces(c, Piece::Bishop) | p.pieces(c, Piece::Queen);
        out[i] = fill::<1, true>(rook, empty & A) & A
            | fill::<1, false>(rook, empty & H) & H
            | fill::<8, true>(rook, empty)
            | fill::<8, false>(rook, empty)
            | fill::<9, true>(bishop, empty & A) & A
            | fill::<7, true>(bishop, empty & H) & H
            | fill::<7, false>(bishop, empty & A) & A
            | fill::<9, false>(bishop, empty & H) & H;
        let pawns = p.pieces(c, Piece::Pawn);
        out[i] |= if i == 0 {
            (pawns << 7) & H | (pawns << 9) & A
        } else {
            (pawns >> 7) & A | (pawns >> 9) & H
        };
        let kings = p.pieces(c, Piece::King);
        let side = (kings << 1) & A | (kings >> 1) & H;
        let triple = kings | side;
        out[i] |= side | triple << 8 | triple >> 8;
        let n = p.pieces(c, Piece::Knight);
        let n1 = (n << 1) & A | (n >> 1) & H;
        let n2 = (n << 2) & 0xfcfcfcfcfcfcfcfc | (n >> 2) & 0x3f3f3f3f3f3f3f3f;
        out[i] |= n1 << 16 | n1 >> 16 | n2 << 8 | n2 >> 8;
    }
    out
}

/// Independently structured auditable ray walker used for all-square checking.
pub fn walk_oracle(p: &Position) -> [u64; 2] {
    let mut result = [0; 2];
    for sq in 0..64 {
        let Some((c, piece)) = p.piece_at(sq) else {
            continue;
        };
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;
        let mut delta = offsets(piece.index(), c.index());
        if matches!(piece, Piece::Rook | Piece::Queen) {
            delta.extend([(1, 0), (-1, 0), (0, 1), (0, -1)]);
        }
        if matches!(piece, Piece::Bishop | Piece::Queen) {
            delta.extend([(1, 1), (-1, 1), (1, -1), (-1, -1)]);
        }
        for (df, dr) in delta {
            let mut nf = f + df;
            let mut nr = r + dr;
            while (0..8).contains(&nf) && (0..8).contains(&nr) {
                let to = (nr * 8 + nf) as u8;
                result[c.index()] |= 1u64 << to;
                if !matches!(piece, Piece::Rook | Piece::Bishop | Piece::Queen)
                    || p.piece_at(to).is_some()
                {
                    break;
                }
                nf += df;
                nr += dr;
            }
        }
    }
    result
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Obligations {
    pub checkers: u64,
    pub pinned: u64,
    pub evasion: u64,
}

fn knight_union(n: u64) -> u64 {
    let n1 = (n << 1) & A | (n >> 1) & H;
    let n2 = (n << 2) & 0xfcfcfcfcfcfcfcfc | (n >> 2) & 0x3f3f3f3f3f3f3f3f;
    n1 << 16 | n1 >> 16 | n2 << 8 | n2 >> 8
}
fn king_union(k: u64) -> u64 {
    let side = (k << 1) & A | (k >> 1) & H;
    let triple = k | side;
    side | triple << 8 | triple >> 8
}
fn pawn_union(p: u64, c: usize) -> u64 {
    if c == 0 {
        (p << 7) & H | (p << 9) & A
    } else {
        (p >> 7) & A | (p >> 9) & H
    }
}
fn local_checkers(planes: &[[u64; 6]; 2], king: u8, c: usize) -> u64 {
    let k = 1u64 << king;
    pawn_union(k, c) & planes[1 - c][0]
        | knight_union(k) & planes[1 - c][1]
        | king_union(k) & planes[1 - c][5]
}
fn local_obligations(planes: &[[u64; 6]; 2], king: u8, c: usize) -> Obligations {
    let checkers = local_checkers(planes, king, c);
    Obligations {
        checkers,
        pinned: 0,
        evasion: match checkers.count_ones() {
            0 => !0,
            1 => checkers,
            _ => 0,
        },
    }
}

/// Conventional target-directed nearest-blocker control for exactly the same
/// checkers/pins/evasion workload as the compressed first-two-line projection.
/// Eight precomputed rays + bit scans avoid an artificially weak square scan.
#[inline(never)]
pub fn obligations_control(p: &Position, t: &Tables, color: Color) -> Obligations {
    let c = color.index();
    let enemy = color.opposite();
    let king = p.king_square(color);
    let mut planes = [[0u64; 6]; 2];
    for c in COLORS {
        for pc in PIECES {
            planes[c.index()][pc.index()] = p.pieces(c, pc);
        }
    }
    let mut result = local_obligations(&planes, king, c);
    let occ = p.occupied();
    let own = planes[c].iter().fold(0, |a, b| a | b);
    let rook = p.pieces(enemy, Piece::Rook) | p.pieces(enemy, Piece::Queen);
    let bishop = p.pieces(enemy, Piece::Bishop) | p.pieces(enemy, Piece::Queen);
    for d in 0..8 {
        let ray = t.rays[king as usize][d];
        let blockers = ray & occ;
        if blockers == 0 {
            continue;
        }
        let increasing = matches!(d, 0 | 2 | 4 | 5);
        let first = if increasing {
            blockers.trailing_zeros()
        } else {
            63 - blockers.leading_zeros()
        } as usize;
        let firstbit = 1u64 << first;
        let emitter = if d < 4 { rook } else { bishop };
        if firstbit & emitter != 0 {
            result.checkers |= firstbit;
            result.evasion &= ray ^ t.rays[first][d];
        } else if firstbit & own != 0 {
            let remaining = t.rays[first][d] & occ;
            if remaining == 0 {
                continue;
            }
            let second = if increasing {
                remaining.trailing_zeros()
            } else {
                63 - remaining.leading_zeros()
            };
            if (1u64 << second) & emitter != 0 {
                result.pinned |= firstbit;
            }
        }
    }
    result
}

/// Compose four boundary summaries at the king. The first friendly obstruction
/// and next matching enemy emitter determine an absolute pin; direct emitters
/// determine check and the joint evasion intersection. Exact for geometric
/// check/pin obligations, not sufficient by itself for complete legal moves.
pub fn obligations(s: &Encoded, t: &Tables, color: usize) -> Obligations {
    obligations_impl(s, t, color, false)
}

pub fn obligations_progressive(s: &Encoded, t: &Tables, color: usize) -> Obligations {
    obligations_impl(s, t, color, true)
}

#[inline(never)]
fn obligations_impl(s: &Encoded, t: &Tables, color: usize, progressive: bool) -> Obligations {
    let king = s.kings[color];
    let local = local_obligations(&s.planes, king, color);
    if progressive && !needs_slider_refinement(s, t, color) {
        return local;
    }
    let mut checkers = local.checkers;
    let mut pinned = 0u64;
    let mut evasion = local.evasion;
    for &(line, ki) in &t.square_lines[king as usize] {
        let code = s.lines[line as usize];
        let near = t.first_two[ki as usize * 256 + (code & 255) as usize];
        let enemy = (code >> (8 + 8 * (1 - color))) & 255;
        for (firstshift, secondshift) in [(0, 4), (8, 12)] {
            let first = ((near >> firstshift) & 15) as usize;
            let second = ((near >> secondshift) & 15) as usize;
            if first >= t.geometry[line as usize].len() {
                continue;
            }
            let sq = t.geometry[line as usize][first];
            if enemy & (1 << first) != 0 {
                checkers |= 1u64 << sq;
                let lo = first.min(ki as usize);
                let hi = first.max(ki as usize);
                let mut ray = 0;
                for index in lo..=hi {
                    if index != ki as usize {
                        ray |= 1u64 << t.geometry[line as usize][index];
                    }
                }
                evasion &= ray;
            } else if s.board[sq as usize] != 0
                && (s.board[sq as usize] - 1) / 6 == color as u8
                && second < t.geometry[line as usize].len()
                && enemy & (1 << second) != 0
            {
                pinned |= 1u64 << sq;
            }
        }
    }
    Obligations {
        checkers,
        pinned,
        evasion,
    }
}

/// Coarse band descriptor can prove *absence* of slider check: no matching
/// enemy emitter shares a king line. Positive answers require blockers to be
/// decoded. This measures progressive refinement rather than pretending that
/// the coarse projection is sufficient for every question.
pub fn needs_slider_refinement(s: &Encoded, t: &Tables, color: usize) -> bool {
    t.square_lines[s.kings[color] as usize]
        .iter()
        .any(|&(line, _)| s.lines[line as usize] >> (8 + 8 * (1 - color)) & 255 != 0)
}

pub fn plane_source_count(position: &Position) -> u32 {
    COLORS
        .iter()
        .flat_map(|&c| {
            PIECES
                .iter()
                .map(move |&p| position.pieces(c, p).count_ones())
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_line_language_and_explicit_avx() {
        let t = Tables::new();
        let mut s = Encoded::new(&Position::startpos(), &t);
        s.lines = [0; 48];
        // All 4^8 valid occupancy/white-source/black-source assignments.
        for raw in 0u32..65536 {
            let mut code = 0;
            let mut x = raw;
            for bit in 0..8 {
                let class = x & 3;
                x >>= 2;
                if class != 0 {
                    code |= 1 << bit;
                }
                if class >= 2 {
                    code |= 1 << (8 + (class - 2) * 8 + bit);
                }
            }
            s.lines[0] = code;
            let mut expect = [0u64; 2];
            for (c, out) in expect.iter_mut().enumerate() {
                for origin in 0..8i32 {
                    if code & (1 << (8 + 8 * c + origin as usize)) == 0 {
                        continue;
                    }
                    for d in [-1, 1] {
                        let mut q = origin + d;
                        while (0..8).contains(&q) {
                            *out |= 1 << q;
                            if code & (1 << q) != 0 {
                                break;
                            }
                            q += d;
                        }
                    }
                }
            }
            assert_eq!(slider_scalar(&s, &t), expect, "raw {raw}");
            if avx_available() {
                assert_eq!(unsafe { slider_avx2(&s, &t) }, expect, "avx raw {raw}");
            }
        }
    }

    #[test]
    fn exhaustive_raw_microblocks_at_every_board_location() {
        let t = Tables::new();
        let mut s = Encoded::new(&Position::startpos(), &t);
        for tile in 0..16 {
            for raw in 0..28561u16 {
                s.micro = [0; 16];
                s.micro[tile] = raw;
                let mut expect = [0u64; 2];
                let mut code = raw as usize;
                for cell in 0..4 {
                    let p = code % 13;
                    code /= 13;
                    if p == 0 {
                        continue;
                    }
                    let color = (p - 1) / 6;
                    let pc = (p - 1) % 6;
                    let f = (tile % 4 * 2 + cell % 2) as i8;
                    let r = (tile / 4 * 2 + cell / 2) as i8;
                    for (df, dr) in offsets(pc, color) {
                        let x = f + df;
                        let y = r + dr;
                        if (0..8).contains(&x) && (0..8).contains(&y) {
                            expect[color] |= 1u64 << (y * 8 + x);
                        }
                    }
                }
                assert_eq!(leapers_micro(&s, &t), expect, "tile {tile} code {raw}");
            }
        }
    }
}
