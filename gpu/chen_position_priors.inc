__device__ uint64_t splitmix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}
__device__ uint64_t hash_key(uint64_t offset) { return splitmix(0x4348454e00000000ull + offset); }
__device__ int pop(uint64_t &bb) {
    const int square=__ffsll(static_cast<unsigned long long>(bb))-1;
    bb &= bb-1; return square;
}
__device__ uint64_t occupied(const ChenRulePosition &p, int color=-1) {
    uint64_t result=0;
    for(int i=0;i<12;++i) if(color<0 || i/6==color) result |= p.boards[i];
    return result;
}
__device__ int piece_at(const ChenRulePosition &p, int square) {
    const uint64_t bit=uint64_t(1)<<square;
    for(int piece=0;piece<12;++piece) if(p.boards[piece]&bit) return piece;
    return -1;
}
__device__ uint64_t leaper(int square, int piece, int color) {
    uint64_t result=0;
    const int x=square%8,y=square/8;
    for(int dx=-2;dx<=2;++dx) for(int dy=-2;dy<=2;++dy) {
        const int ax=dx<0?-dx:dx, ay=dy<0?-dy:dy;
        const bool allowed=piece==0 ? (ax==1 && dy==(color?-1:1)) :
            piece==1 ? (ax*ay==2 && ax+ay==3) : (ax<=1 && ay<=1 && ax+ay>0);
        if(allowed && x+dx>=0 && x+dx<8 && y+dy>=0 && y+dy<8)
            result |= uint64_t(1)<<((y+dy)*8+x+dx);
    }
    return result;
}
__device__ uint64_t sliding(int square, uint64_t occ, bool diagonal, bool straight) {
    uint64_t result=0;
    for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) {
        if((!dx&&!dy) || (dx&&dy ? !diagonal : !straight)) continue;
        int x=square%8+dx,y=square/8+dy;
        while(x>=0 && x<8 && y>=0 && y<8) {
            const uint64_t bit=uint64_t(1)<<(y*8+x); result |= bit;
            if(occ&bit) break;
            x+=dx; y+=dy;
        }
    }
    return result;
}
__device__ uint64_t attackers(const ChenRulePosition &p, int square, int color) {
    const int base=color*6;
    const uint64_t occ=occupied(p);
    return (leaper(square,0,1-color)&p.boards[base]) |
        (leaper(square,1,color)&p.boards[base+1]) |
        (leaper(square,5,color)&p.boards[base+5]) |
        (sliding(square,occ,true,false)&(p.boards[base+2]|p.boards[base+4])) |
        (sliding(square,occ,false,true)&(p.boards[base+3]|p.boards[base+4]));
}
__device__ bool in_check(const ChenRulePosition &p,int color) {
    const int square=__ffsll(static_cast<unsigned long long>(p.boards[color*6+5]))-1;
    return attackers(p,square,1-color)!=0;
}
__device__ int effective_ep(const ChenRulePosition &p) {
    if(p.ep_square<0) return -1;
    const int color=int(p.side),target=p.ep_square;
    uint64_t candidates=leaper(target,0,1-color)&p.boards[color*6];
    while(candidates) {
        const int from=pop(candidates),captured=target+(color?8:-8);
        ChenRulePosition scratch=p;
        scratch.boards[color*6] ^= (uint64_t(1)<<from)|(uint64_t(1)<<target);
        scratch.boards[(1-color)*6] ^= uint64_t(1)<<captured;
        if(!in_check(scratch,color)) return target%8;
    }
    return -1;
}
__device__ uint64_t recompute_hash(const ChenRulePosition &p) {
    uint64_t hash=hash_key(768+p.castling);
    if(p.side) hash ^= hash_key(792);
    const int ep=effective_ep(p);
    if(ep>=0) hash ^= hash_key(784+ep);
    for(int piece=0;piece<12;++piece) {
        uint64_t bb=p.boards[piece];
        while(bb) hash ^= hash_key(piece*64+pop(bb));
    }
    return hash;
}
__device__ bool valid_position(const ChenRulePosition &p) {
    if(p.side>1 || p.castling>15 || p.reserved || p.halfmove>65535 || !p.fullmove ||
        p.ep_square<-1 || p.ep_square>=64) return false;
    uint64_t occ=0;
    for(int i=0;i<12;++i) { if(occ&p.boards[i]) return false; occ |= p.boards[i]; }
    if(__popcll(p.boards[5])!=1 || __popcll(p.boards[11])!=1 ||
        ((p.boards[0]|p.boards[6])&0xff000000000000ffull)) return false;
    const int white=__ffsll(static_cast<unsigned long long>(p.boards[5]))-1;
    if(leaper(white,5,0)&p.boards[11]) return false;
    for(int color=0;color<2;++color) {
        const int base=color?56:0;
        if((p.castling&(3u<<(color*2))) && piece_at(p,base+4)!=color*6+5) return false;
        if((p.castling&(1u<<(color*2))) && piece_at(p,base+7)!=color*6+3) return false;
        if((p.castling&(2u<<(color*2))) && piece_at(p,base)!=color*6+3) return false;
    }
    if(p.ep_square>=0) {
        const int ep=p.ep_square, pawn=ep+(p.side?8:-8),origin=ep+(p.side?-8:8);
        if(ep/8!=(p.side?2:5) || piece_at(p,ep)>=0 ||
            piece_at(p,pawn)!=int((1-p.side)*6) || piece_at(p,origin)>=0) return false;
    }
    return p.hash==recompute_hash(p);
}
