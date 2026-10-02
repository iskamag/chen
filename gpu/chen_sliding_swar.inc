// Branchless sliding attacks for a1=bit0 bitboards. Include inside the rules
// device namespace. Pure occupancy/origin operation: no tree, ABI, or scheduler
// state. The final shift includes the first blocker and excludes the origin.
template<int Shift, bool Increasing>
__device__ __forceinline__ uint64_t chen_sliding_fill(uint64_t generator, uint64_t propagator) {
    static_assert(Shift > 0 && Shift <= 9, "board ray shift outside supported range");
    if constexpr (Increasing) {
        generator |= propagator & (generator << Shift);
        propagator &= propagator << Shift;
        generator |= propagator & (generator << (2 * Shift));
        propagator &= propagator << (2 * Shift);
        generator |= propagator & (generator << (4 * Shift));
        return generator << Shift;
    } else {
        generator |= propagator & (generator >> Shift);
        propagator &= propagator >> Shift;
        generator |= propagator & (generator >> (2 * Shift));
        propagator &= propagator >> (2 * Shift);
        generator |= propagator & (generator >> (4 * Shift));
        return generator >> Shift;
    }
}

__device__ __forceinline__ uint64_t sliding(int square, uint64_t occupied,
                                           bool diagonal, bool straight) {
    constexpr uint64_t NOT_A_FILE = 0xfefefefefefefefeull;
    constexpr uint64_t NOT_H_FILE = 0x7f7f7f7f7f7f7f7full;
    const uint64_t generator = uint64_t(1) << square;
    const uint64_t empty = ~occupied;
    const uint64_t east = empty & NOT_A_FILE;
    const uint64_t west = empty & NOT_H_FILE;
    uint64_t attacks = 0;
    if (straight) {
        attacks |= chen_sliding_fill<1, true>(generator, east) & NOT_A_FILE;
        attacks |= chen_sliding_fill<1, false>(generator, west) & NOT_H_FILE;
        attacks |= chen_sliding_fill<8, true>(generator, empty);
        attacks |= chen_sliding_fill<8, false>(generator, empty);
    }
    if (diagonal) {
        attacks |= chen_sliding_fill<9, true>(generator, east) & NOT_A_FILE;
        attacks |= chen_sliding_fill<7, true>(generator, west) & NOT_H_FILE;
        attacks |= chen_sliding_fill<7, false>(generator, east) & NOT_A_FILE;
        attacks |= chen_sliding_fill<9, false>(generator, west) & NOT_H_FILE;
    }
    return attacks;
}
