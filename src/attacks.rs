//! Experimental occupancy-only sliding attacks, independent of search and rules.
//!
//! Squares use a1 = bit 0. Callers supply an origin in 0..64; attacks include the
//! first occupied square in each direction and exclude the origin. Three
//! Kogge–Stone propagation stages cover the maximum seven-square board ray.
//! This module is admitted through a controlled candidate build before changing
//! the production core's implementation.

const NOT_A_FILE: u64 = 0xfefe_fefe_fefe_fefe;
const NOT_H_FILE: u64 = 0x7f7f_7f7f_7f7f_7f7f;

#[inline(always)]
fn fill<const SHIFT: u32, const INCREASING: bool>(mut g: u64, mut p: u64) -> u64 {
    if INCREASING {
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

#[inline(always)]
pub fn rook(square: u8, occupied: u64) -> u64 {
    debug_assert!(square < 64);
    let g = 1u64 << square;
    let empty = !occupied;
    (fill::<1, true>(g, empty & NOT_A_FILE) & NOT_A_FILE)
        | (fill::<1, false>(g, empty & NOT_H_FILE) & NOT_H_FILE)
        | fill::<8, true>(g, empty)
        | fill::<8, false>(g, empty)
}

#[inline(always)]
pub fn bishop(square: u8, occupied: u64) -> u64 {
    debug_assert!(square < 64);
    let g = 1u64 << square;
    let empty = !occupied;
    let east = empty & NOT_A_FILE;
    let west = empty & NOT_H_FILE;
    (fill::<9, true>(g, east) & NOT_A_FILE)
        | (fill::<7, true>(g, west) & NOT_H_FILE)
        | (fill::<7, false>(g, east) & NOT_A_FILE)
        | (fill::<9, false>(g, west) & NOT_H_FILE)
}
