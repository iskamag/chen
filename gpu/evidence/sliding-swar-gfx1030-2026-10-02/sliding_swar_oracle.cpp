// Exact host enumeration of relevant ray occupancies, using the actual candidate
// source. The oracle deliberately walks squares rather than reusing fill logic.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __device__
#define __forceinline__ inline
#include "chen_sliding_swar.inc"
uint64_t walk(int s,uint64_t occ,bool diagonal,bool straight) {
    uint64_t result=0;
    for(int df=-1;df<=1;++df) for(int dr=-1;dr<=1;++dr) {
        if((!df&&!dr)||((df&&dr)?!diagonal:!straight)) continue;
        int f=s%8+df,r=s/8+dr;
        while(f>=0&&f<8&&r>=0&&r<8) {
            uint64_t bit=1ull<<(r*8+f);result|=bit;if(occ&bit)break;f+=df;r+=dr;
        }
    }
    return result;
}
int main() {
    uint64_t checked=0;
    for(int diagonal=0;diagonal<2;++diagonal) for(int square=0;square<64;++square) {
        uint64_t relevant=walk(square,0,diagonal,!diagonal);
        std::vector<int> bits;
        for(int s=0;s<64;++s) if(relevant&(1ull<<s))bits.push_back(s);
        for(unsigned subset=0;subset<(1u<<bits.size());++subset) {
            uint64_t occ=0;
            for(unsigned b=0;b<bits.size();++b)if(subset&(1u<<b))occ|=1ull<<bits[b];
            for(int origin=0;origin<2;++origin) {
                uint64_t board=occ|(origin?(1ull<<square):0);
                if(sliding(square,board,diagonal,!diagonal)!=walk(square,board,diagonal,!diagonal)) {
                    fprintf(stderr,"mismatch square=%d diagonal=%d occupancy=%llx\n",square,diagonal,(unsigned long long)board);return 1;
                }
                ++checked;
            }
        }
    }
    uint64_t seed=0xc4e103020261002ull;
    for(int i=0;i<100000;++i) {
        seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
        int s=i%64;
        if(sliding(s,seed,true,true)!=walk(s,seed,true,true)||sliding(s,seed,false,false)!=0)return 1;
        checked+=2;
    }
    printf("{\"status\":\"passed\",\"checks\":%llu,\"relevant_ray_occupancy_enumeration\":true,\"both_origin_bits\":true,\"queen_random_checks\":100000}\n",(unsigned long long)checked);
}
