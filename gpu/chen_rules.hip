#include "chen_rules.h"
#include "chen_replay.h"
#include "chen_teacher.h"
#include "chen_external.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace {
constexpr uint64_t NONE = UINT64_MAX;
constexpr uint32_t CHECK=1, INSUFFICIENT=2, CLASSIFIED=4, TERMINAL=8, SCORED=16, EXPANDED=32;
constexpr int MATE=30000, MAX_EVAL=29000;
struct Node {
    uint64_t parent, first_child, best;
    uint32_t child_count, ply, flags;
    int32_t score;
    ChenRuleMove move;
    uint32_t reserved;
};
struct Summary { uint64_t next_count; uint32_t error, reserved; };
__device__ void device_error(uint32_t *error, uint32_t code) { atomicCAS(error, 0u, code); }
#include "chen_position_priors.inc"
template<bool Hash> __device__ void toggle(ChenRulePosition &p,int piece,int square) {
    p.boards[piece] ^= uint64_t(1)<<square;
    if constexpr(Hash) p.hash ^= hash_key(piece*64+square);
}
template<bool Hash> __device__ ChenRulePosition apply(const ChenRulePosition &input,ChenRuleMove move) {
    ChenRulePosition p=input;
    const int color=int(p.side),piece=piece_at(p,move.from),kind=piece%6;
    int captured=piece_at(p,move.to);
    if constexpr(Hash) {
        p.hash ^= hash_key(768+p.castling);
        const int ep=effective_ep(p); if(ep>=0) p.hash ^= hash_key(784+ep);
    }
    if(kind==0 && p.ep_square==move.to && captured<0) {
        toggle<Hash>(p,(1-color)*6,move.to+(color?8:-8)); captured=(1-color)*6;
    } else if(captured>=0) toggle<Hash>(p,captured,move.to);
    toggle<Hash>(p,piece,move.from);
    toggle<Hash>(p,color*6+(move.promotion?move.promotion:kind),move.to);
    if(kind==5 && (int(move.from)-int(move.to)==2 || int(move.to)-int(move.from)==2)) {
        const int rook_from=move.to>move.from?move.from+3:move.from-4;
        const int rook_to=move.to>move.from?move.from+1:move.from-1;
        toggle<Hash>(p,color*6+3,rook_from); toggle<Hash>(p,color*6+3,rook_to);
    }
    if(kind==5) p.castling &= ~(3u<<(color*2));
    const int squares[2]={move.from,move.to};
    for(int square:squares) {
        if(square==0) p.castling &= ~2u;
        if(square==7) p.castling &= ~1u;
        if(square==56) p.castling &= ~8u;
        if(square==63) p.castling &= ~4u;
    }
    p.ep_square=kind==0 && (int(move.from)-int(move.to)==16 || int(move.to)-int(move.from)==16)
        ? (move.from+move.to)/2 : -1;
    p.halfmove=kind==0 || captured>=0 ? 0 : (p.halfmove<65535?p.halfmove+1:65535);
    if(color && p.fullmove<UINT32_MAX) ++p.fullmove;
    p.side=1-color;
    if constexpr(Hash) {
        p.hash ^= hash_key(792)^hash_key(768+p.castling);
        const int ep=effective_ep(p); if(ep>=0) p.hash ^= hash_key(784+ep);
    }
    return p;
}
__device__ bool insufficient(const ChenRulePosition &p) {
    if(p.boards[0]|p.boards[3]|p.boards[4]|p.boards[6]|p.boards[9]|p.boards[10]) return false;
    const uint64_t knights=p.boards[1]|p.boards[7],bishops=p.boards[2]|p.boards[8];
    if(__popcll(knights|bishops)<=1) return true;
    if(knights) return false;
    return !(bishops&0xaa55aa55aa55aa55ull) || !(bishops&0x55aa55aa55aa55aaull);
}
__device__ void candidate(const ChenRulePosition &p,int from,int to,int promotion,
    ChenRuleMove *moves,uint32_t &count,uint32_t *error) {
    const ChenRuleMove move{uint8_t(from),uint8_t(to),uint8_t(promotion),0};
    const ChenRulePosition next=apply<false>(p,move);
    if(in_check(next,p.side)) return;
    if(count>=CHEN_RULES_MAX_MOVES) { device_error(error,2); return; }
    if(moves) moves[count]=move;
    ++count;
}
__device__ void pawn_candidate(const ChenRulePosition &p,int from,int to,
    ChenRuleMove *moves,uint32_t &count,uint32_t *error) {
    if(to/8==0 || to/8==7) {
        for(int promotion=4;promotion>=1;--promotion) candidate(p,from,to,promotion,moves,count,error);
    } else candidate(p,from,to,0,moves,count,error);
}
__device__ uint32_t legal_moves(const ChenRulePosition &p,ChenRuleMove *moves,uint32_t *error) {
    uint32_t count=0; const int color=int(p.side),base=color*6;
    const uint64_t own=occupied(p,color),occ=occupied(p),enemy=occupied(p,1-color)&~p.boards[(1-color)*6+5];
    const uint64_t allowed=~own&~p.boards[(1-color)*6+5];
    uint64_t pawns=p.boards[base];
    while(pawns) {
        const int from=pop(pawns),to=from+(color?-8:8);
        if(!(occ&(uint64_t(1)<<to))) {
            pawn_candidate(p,from,to,moves,count,error);
            if(from/8==(color?6:1)) {
                const int twice=from+(color?-16:16);
                if(!(occ&(uint64_t(1)<<twice))) candidate(p,from,twice,0,moves,count,error);
            }
        }
        uint64_t captures=leaper(from,0,color)&enemy;
        if(p.ep_square>=0) captures |= leaper(from,0,color)&(uint64_t(1)<<p.ep_square);
        while(captures) pawn_candidate(p,from,pop(captures),moves,count,error);
    }
    for(int piece=1;piece<6;++piece) {
        uint64_t bb=p.boards[base+piece];
        while(bb) {
            const int from=pop(bb);
            uint64_t targets=(piece==1 || piece==5 ? leaper(from,piece,color) :
                sliding(from,occ,piece==2 || piece==4,piece==3 || piece==4))&allowed;
            while(targets) candidate(p,from,pop(targets),0,moves,count,error);
        }
    }
    if(!in_check(p,color)) {
        const int home=color?56:0;
        if((p.castling&(1u<<(color*2))) && !(occ&((uint64_t(1)<<(home+5))|(uint64_t(1)<<(home+6)))) &&
            !attackers(p,home+5,1-color) && !attackers(p,home+6,1-color))
            candidate(p,home+4,home+6,0,moves,count,error);
        if((p.castling&(2u<<(color*2))) && !(occ&((uint64_t(1)<<(home+1))|(uint64_t(1)<<(home+2))|(uint64_t(1)<<(home+3)))) &&
            !attackers(p,home+3,1-color) && !attackers(p,home+2,1-color))
            candidate(p,home+4,home+2,0,moves,count,error);
    }
    return count;
}
__device__ int evaluate(const ChenRulePosition &p,uint32_t mode) {
    constexpr int values[6]={100,320,330,500,900,0},phases[6]={0,1,1,2,4,0};
    int mg=0,eg=0,phase=0;
    for(int color=0;color<2;++color) {
        const int sign=color?-1:1;
        for(int piece=0;piece<6;++piece) {
            uint64_t bb=p.boards[color*6+piece]; const int n=__popcll(bb);
            if(!mode) { mg+=sign*values[piece]*n; continue; }
            phase+=phases[piece]*n;
            while(bb) {
                const int square=pop(bb),file=square%8,rank=color?7-square/8:square/8;
                const int fx=2*file-7,ry=2*rank-7,center=14-(fx<0?-fx:fx)-(ry<0?-ry:ry);
                int m=0,e=0;
                switch(piece) {
                    case 0:m=rank*8+center*2;e=rank*14+center;break;
                    case 1:m=center*6;e=center*4;break;
                    case 2:m=center*3+rank*2;e=center*3;break;
                    case 3:m=rank*2+(rank==6?18:0);e=center;break;
                    case 4:m=center;e=center*2;break;
                    case 5:m=-center*5-rank*4+((rank==0&&(file==2||file==6))?35:0);e=center*6;break;
                }
                mg+=sign*(values[piece]+m);eg+=sign*(values[piece]+e);
            }
        }
        if(mode && __popcll(p.boards[color*6+2])>=2) { mg+=sign*25;eg+=sign*40; }
    }
    if(phase>24) phase=24;
    int score=mode?(mg*phase+eg*(24-phase))/24:mg;
    if(p.side) score=-score;
    return score<-MAX_EVAL?-MAX_EVAL:score>MAX_EVAL?MAX_EVAL:score;
}

__global__ void generic_legal(const ChenRulePosition *positions,ChenRuleMove *moves,
    uint32_t *counts,uint32_t *flags,uint64_t count,uint32_t *error) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const ChenRulePosition p=positions[i];
    if(!valid_position(p)) { device_error(error,1); return; }
    counts[i]=legal_moves(p,moves+i*CHEN_RULES_MAX_MOVES,error);
    flags[i]=(in_check(p,p.side)?CHECK:0)|(insufficient(p)?INSUFFICIENT:0);
}
__global__ void generic_make(const ChenRulePosition *positions,const ChenRuleMove *requested,
    ChenRuleMove *moves,ChenRulePosition *out,uint64_t count,uint32_t *error) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const ChenRulePosition p=positions[i];
    if(!valid_position(p)) { device_error(error,1); return; }
    const uint32_t n=legal_moves(p,moves+i*CHEN_RULES_MAX_MOVES,error);
    const ChenRuleMove move=requested[i];
    for(uint32_t j=0;j<n;++j) {
        const ChenRuleMove legal=moves[i*CHEN_RULES_MAX_MOVES+j];
        if(move.from==legal.from && move.to==legal.to && move.promotion==legal.promotion && !move.reserved) {
            out[i]=apply<true>(p,move); return;
        }
    }
    device_error(error,3);
}
__global__ void reset_root(const ChenRulePosition *positions,Node *nodes,uint32_t *error) {
    if(!valid_position(positions[0])) { device_error(error,1); return; }
    nodes[0]=Node{NONE,0,NONE,0,0,0,0,ChenRuleMove{0,0,0,0},0};
}
__device__ bool repeated(const ChenRulePosition *positions,const Node *nodes,uint64_t index,
    const uint64_t *history,uint64_t history_count) {
    const uint64_t key=positions[index].hash;
    uint32_t remaining=positions[index].halfmove+1,count=0;
    while(remaining) {
        if(positions[index].hash==key) ++count;
        --remaining;
        if(nodes[index].parent==NONE) break;
        index=nodes[index].parent;
    }
    while(remaining && history_count) {
        if(history[--history_count]==key) ++count;
        --remaining;
    }
    return count>=3;
}
__global__ void classify_kernel(const ChenRulePosition *positions,Node *nodes,ChenRuleMove *moves,
    uint32_t *counts,uint64_t start,uint64_t count,uint32_t adjudicate,
    const uint64_t *history,uint64_t history_count,unsigned long long *classified,uint32_t *error) {
    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(offset>=count) return; const uint64_t i=start+offset;
    Node &node=nodes[i]; const ChenRulePosition p=positions[i];
    if(node.flags&EXPANDED) { device_error(error,4); return; }
    if(node.flags&CLASSIFIED) return;
    counts[i]=legal_moves(p,moves+i*CHEN_RULES_MAX_MOVES,error);
    node.flags=(in_check(p,p.side)?CHECK:0)|(insufficient(p)?INSUFFICIENT:0)|CLASSIFIED;
    atomicAdd(classified,1ull);
    if(!counts[i]) {
        node.score=(node.flags&CHECK)?-MATE+int(node.ply):0;
        node.flags |= TERMINAL|SCORED;
    } else if(adjudicate && (p.halfmove>=100 || (node.flags&INSUFFICIENT) ||
        repeated(positions,nodes,i,history,history_count))) {
        node.score=0;node.flags |= TERMINAL|SCORED;
    }
}
__global__ void prefix_kernel(Node *nodes,const uint32_t *counts,uint64_t start,uint64_t count,
    uint64_t used,uint64_t capacity,Summary *summary) {
    uint64_t total=0;
    for(uint64_t j=0;j<count;++j) {
        const Node &node=nodes[start+j];
        if(!(node.flags&TERMINAL)) total+=counts[start+j];
        if(total>capacity-used) { device_error(&summary->error,2); return; }
    }
    summary->next_count=total; uint64_t cursor=used;
    for(uint64_t j=0;j<count;++j) {
        Node &node=nodes[start+j];
        node.first_child=cursor; node.child_count=(node.flags&TERMINAL)?0:counts[start+j];
        cursor+=node.child_count;
    }
}
__global__ void expansion_preflight(const Node *nodes,uint64_t start,uint64_t count,uint32_t *error) {
    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(offset>=count) return;
    const uint32_t flags=nodes[start+offset].flags;
    if((flags&EXPANDED) || ((flags&SCORED) && !(flags&TERMINAL))) device_error(error,4);
}
__global__ void expand_kernel(ChenRulePosition *positions,Node *nodes,const ChenRuleMove *moves,
    uint64_t start,uint64_t count) {
    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(offset>=count) return; const uint64_t i=start+offset;
    const Node parent=nodes[i];
    for(uint32_t j=0;j<parent.child_count;++j) {
        const uint64_t child=parent.first_child+j;
        const ChenRuleMove move=moves[i*CHEN_RULES_MAX_MOVES+j];
        positions[child]=apply<true>(positions[i],move);
        nodes[child]=Node{i,0,NONE,0,parent.ply+1,0,0,move,0};
    }
    if(parent.child_count) nodes[i].flags |= EXPANDED;
}
__global__ void evaluate_kernel_rules(const ChenRulePosition *positions,Node *nodes,uint64_t start,
    uint64_t count,uint32_t mode,unsigned long long *evaluated,uint32_t *error) {
    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(offset>=count) return; const uint64_t i=start+offset;
    Node &node=nodes[i];
    if(node.flags&EXPANDED) { device_error(error,4); return; }
    if(!(node.flags&SCORED)) {
        node.score=evaluate(positions[i],mode);node.flags|=SCORED;atomicAdd(evaluated,1ull);
    }
}
__global__ void reduce_kernel(Node *nodes,uint64_t start,uint64_t count,uint32_t *error) {
    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(offset>=count) return; const uint64_t i=start+offset;
    Node &node=nodes[i];
    if(node.flags&TERMINAL) return;
    if(!(node.flags&EXPANDED) || !node.child_count) { device_error(error,4); return; }
    int best=-MATE-1;uint64_t best_child=NONE;
    for(uint32_t j=0;j<node.child_count;++j) {
        const uint64_t child=node.first_child+j;
        if(!(nodes[child].flags&SCORED)) { device_error(error,4); return; }
        const int score=-nodes[child].score;
        if(score>best) { best=score;best_child=child; }
    }
    node.score=best;node.best=best_child;node.flags|=SCORED;
}
__global__ void reduction_preflight(const Node *nodes,uint64_t start,uint64_t count,
    uint64_t used,uint32_t *error) {
    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(offset>=count) return;
    const Node node=nodes[start+offset];
    if(node.ply!=nodes[start].ply) { device_error(error,4);return; }
    if(node.flags&TERMINAL) {
        if(!(node.flags&SCORED)) device_error(error,4);
        return;
    }
    if(!(node.flags&EXPANDED) || !node.child_count) { device_error(error,4);return; }
    for(uint32_t j=0;j<node.child_count;++j) {
        const uint64_t child=node.first_child+j;
        if(child>=used || (child>=start && child<start+count) || !(nodes[child].flags&SCORED)) {
            device_error(error,4);return;
        }
    }
}
__global__ void finish_kernel(const Node *nodes,ChenRulesResult *result,ChenRuleMove *pv,
    uint64_t used,uint32_t *error) {
    if(!(nodes[0].flags&SCORED)) { device_error(error,4); return; }
    result->score=nodes[0].score;result->pv_length=0;uint64_t index=0;
    while(nodes[index].best!=NONE) {
        index=nodes[index].best;
        if(index>=used || result->pv_length>=used) { device_error(error,4); return; }
        pv[result->pv_length++]=nodes[index].move;
    }
}
}

struct ChenRulesContext {
    int device=0,mode=-1,evaluator=-1;uint64_t capacity=0,used=0,history_count=0,root_epoch=0;double kernel_total_ms=0;
    ChenRulePosition *positions=nullptr,*scratch=nullptr,*host_positions=nullptr;
    ChenRuleMove *moves=nullptr,*requested=nullptr,*host_moves=nullptr;
    uint32_t *counts=nullptr,*flags=nullptr,*host_counts=nullptr,*host_flags=nullptr;
    Node *nodes=nullptr;uint64_t *history=nullptr;
    unsigned long long *classified=nullptr,*evaluated=nullptr;
    Summary *summary=nullptr;ChenRulesResult *result=nullptr;
    hipStream_t stream=nullptr;hipEvent_t begin=nullptr,end=nullptr;
    hipEvent_t transfers[4]={};
    ChenExternalEvaluate external_callback=nullptr;
    void *external_owner=nullptr;
    uint64_t external_max_batch=0;
    // Independently owned arena extensions register private allocation bounds.
    // Rules contexts own none of these allocations; the extension retires them.
    const void *extra_private[4]={};
    uint64_t extra_private_bytes[4]={};
};
namespace {
int32_t fail(char *error,uint64_t capacity,const char *message,int32_t code=-1) {
    if(error&&capacity) std::snprintf(error,size_t(capacity),"%s",message);
    return code;
}
void clear_error(char *error,uint64_t capacity) { if(error&&capacity) *error='\0'; }
int32_t hip_fail(char *error,uint64_t capacity,hipError_t code) { return fail(error,capacity,hipGetErrorString(code)); }
int32_t prepare(ChenRulesContext *ctx,char *error,uint64_t capacity) {
    clear_error(error,capacity);
    if(!ctx) return fail(error,capacity,"null rules context");
    hipError_t code=hipSetDevice(ctx->device);
    if(code==hipSuccess) code=hipMemsetAsync(ctx->summary,0,sizeof(Summary),ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->begin,ctx->stream);
    return code==hipSuccess?0:hip_fail(error,capacity,code);
}
int32_t complete(ChenRulesContext *ctx,double *ms,char *error,uint64_t capacity) {
    hipError_t code=hipGetLastError();
    if(code==hipSuccess) code=hipEventRecord(ctx->end,ctx->stream);
    const hipError_t sync=hipStreamSynchronize(ctx->stream);
    if(code==hipSuccess) code=sync;
    Summary summary{};float elapsed=0;
    if(code==hipSuccess) code=hipMemcpy(&summary,ctx->summary,sizeof(summary),hipMemcpyDeviceToHost);
    if(code==hipSuccess) code=hipEventElapsedTime(&elapsed,ctx->begin,ctx->end);
    if(code!=hipSuccess) return hip_fail(error,capacity,code);
    if(ms) *ms=elapsed;
    ctx->kernel_total_ms+=elapsed;
    switch(summary.error) {
        case 0:return 0;
        case 1:return fail(error,capacity,"invalid rules position or mismatched hash");
        case 2:return fail(error,capacity,"rules arena/move capacity exceeded",CHEN_RULES_CAPACITY);
        case 3:return fail(error,capacity,"requested move is not legal");
        case 5:return fail(error,capacity,"external evaluator score missing or outside +/-29000");
        default:return fail(error,capacity,"invalid resident tree stage or incomplete child scores");
    }
}
int32_t statistics(ChenRulesContext *ctx,ChenRulesStats *stats,double,char *error,uint64_t cap) {
    if(!stats) return fail(error,cap,"null rules statistics");
    unsigned long long classified=0,evaluated=0;
    hipError_t code=hipMemcpy(&classified,ctx->classified,sizeof(classified),hipMemcpyDeviceToHost);
    if(code==hipSuccess) code=hipMemcpy(&evaluated,ctx->evaluated,sizeof(evaluated),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    *stats=ChenRulesStats{ctx->used,uint64_t(classified),uint64_t(evaluated),0,0,ctx->kernel_total_ms}; return 0;
}
int32_t transfer_failure(ChenRulesContext *ctx,hipError_t code,char *error,uint64_t cap) {
    (void)hipStreamSynchronize(ctx->stream);
    return hip_fail(error,cap,code);
}
int32_t complete_transfer(ChenRulesContext *ctx,ChenTimings *timing,char *error,uint64_t cap) {
    const hipError_t sync=hipStreamSynchronize(ctx->stream);
    if(sync!=hipSuccess) return hip_fail(error,cap,sync);
    Summary summary{};hipError_t code=hipMemcpy(&summary,ctx->summary,sizeof(summary),hipMemcpyDeviceToHost);
    float spans[3]={};
    for(int i=0;i<3&&code==hipSuccess;++i)
        code=hipEventElapsedTime(&spans[i],ctx->transfers[i],ctx->transfers[i+1]);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    if(summary.error==2) return fail(error,cap,"rules move capacity exceeded",CHEN_RULES_CAPACITY);
    if(summary.error==1) return fail(error,cap,"invalid rules position or mismatched hash");
    if(summary.error==3) return fail(error,cap,"requested move is not legal");
    if(summary.error) return fail(error,cap,"rules device validation failed");
    timing->h2d_ms=spans[0];timing->kernel_ms=spans[1];timing->d2h_ms=spans[2];return 0;
}
bool range_valid(ChenRulesContext *ctx,uint64_t start,uint64_t count) {
    return ctx && ctx->used && start<=ctx->used && count<=ctx->used-start;
}
int32_t lock_mode(ChenRulesContext *ctx,int mode,char *error,uint64_t cap) {
    if(ctx->mode>=0 && ctx->mode!=mode) return fail(error,cap,"resident perft/search mode cannot change before reset");
    ctx->mode=mode;return 0;
}
dim3 grid(uint64_t count) { return dim3(uint32_t((count+63)/64)); }
}

extern "C" uint32_t chen_rules_abi_version() { return 1; }
extern "C" void chen_rules_destroy(ChenRulesContext *ctx) {
    if(!ctx) return;
    (void)hipSetDevice(ctx->device);
    if(ctx->stream) (void)hipStreamSynchronize(ctx->stream);
    if(ctx->begin) (void)hipEventDestroy(ctx->begin);
    if(ctx->end) (void)hipEventDestroy(ctx->end);
    for(auto event:ctx->transfers) if(event) (void)hipEventDestroy(event);
    void *device_buffers[]={ctx->positions,ctx->scratch,ctx->moves,ctx->requested,ctx->counts,
        ctx->flags,ctx->nodes,ctx->history,ctx->classified,ctx->evaluated,ctx->summary,ctx->result};
    for(void *buffer:device_buffers) if(buffer) (void)hipFree(buffer);
    void *host_buffers[]={ctx->host_positions,ctx->host_moves,ctx->host_counts,ctx->host_flags};
    for(void *buffer:host_buffers) if(buffer) (void)hipHostFree(buffer);
    if(ctx->stream) (void)hipStreamDestroy(ctx->stream);
    delete ctx;
}
extern "C" ChenRulesContext *chen_rules_create(int32_t device,uint64_t capacity,char *error,uint64_t error_capacity) {
    clear_error(error,error_capacity);
    if(std::getenv("HSA_OVERRIDE_GFX_VERSION")) {
        fail(error,error_capacity,"architecture overrides are forbidden"); return nullptr;
    }
    if(!capacity || capacity>uint64_t(INT32_MAX) || capacity>std::numeric_limits<size_t>::max()/(CHEN_RULES_MAX_MOVES*sizeof(ChenRuleMove))) {
        fail(error,error_capacity,"invalid rules arena capacity");return nullptr;
    }
    auto *ctx=new(std::nothrow) ChenRulesContext;
    if(!ctx) { fail(error,error_capacity,"rules context allocation failed");return nullptr; }
    ctx->device=device;ctx->capacity=capacity;
    hipDeviceProp_t prop{};hipError_t code=hipSetDevice(device);
    if(code==hipSuccess) code=hipGetDeviceProperties(&prop,device);
    if(code!=hipSuccess) { hip_fail(error,error_capacity,code);chen_rules_destroy(ctx);return nullptr; }
    if(std::strncmp(prop.gcnArchName,"gfx1030",7) || (prop.gcnArchName[7] && prop.gcnArchName[7]!=':')) {
        fail(error,error_capacity,"rules build requires an actual gfx1030 device");chen_rules_destroy(ctx);return nullptr;
    }
    code=hipStreamCreateWithFlags(&ctx->stream,hipStreamNonBlocking);
    if(code==hipSuccess) code=hipEventCreate(&ctx->begin);
    if(code==hipSuccess) code=hipEventCreate(&ctx->end);
    for(int i=0;i<4&&code==hipSuccess;++i) code=hipEventCreate(&ctx->transfers[i]);
#define DEVICE_ALLOC(field,n) if(code==hipSuccess) code=hipMalloc(&ctx->field,(n)*sizeof(*ctx->field))
#define HOST_ALLOC(field,n) if(code==hipSuccess) code=hipHostMalloc(&ctx->field,(n)*sizeof(*ctx->field))
    DEVICE_ALLOC(positions,capacity);DEVICE_ALLOC(scratch,capacity);
    DEVICE_ALLOC(moves,capacity*CHEN_RULES_MAX_MOVES);DEVICE_ALLOC(requested,capacity);
    DEVICE_ALLOC(counts,capacity);DEVICE_ALLOC(flags,capacity);DEVICE_ALLOC(nodes,capacity);
    DEVICE_ALLOC(history,65535);DEVICE_ALLOC(classified,1);DEVICE_ALLOC(evaluated,1);
    DEVICE_ALLOC(summary,1);DEVICE_ALLOC(result,1);
    HOST_ALLOC(host_positions,capacity);HOST_ALLOC(host_moves,capacity*CHEN_RULES_MAX_MOVES);
    HOST_ALLOC(host_counts,capacity);HOST_ALLOC(host_flags,capacity);
#undef DEVICE_ALLOC
#undef HOST_ALLOC
    if(code==hipSuccess) code=hipMemset(ctx->classified,0,sizeof(*ctx->classified));
    if(code==hipSuccess) code=hipMemset(ctx->evaluated,0,sizeof(*ctx->evaluated));
    if(code!=hipSuccess) { hip_fail(error,error_capacity,code);chen_rules_destroy(ctx);return nullptr; }
    return ctx;
}
extern "C" int32_t chen_rules_legal(ChenRulesContext *ctx,const ChenRulePosition *positions,uint64_t count,
    ChenRuleMove *moves,uint32_t *counts,uint32_t *flags,ChenTimings *timing,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !timing || (count&&(!positions||!moves||!counts||!flags))) return fail(error,cap,"null rules legal buffer");
    if(count>ctx->capacity) return fail(error,cap,"rules batch exceeds capacity",CHEN_RULES_CAPACITY);
    if(ctx->root_epoch==UINT64_MAX) return fail(error,cap,"root epoch exhausted; create a new context");
    *timing=ChenTimings{0,0,0,count,ctx->capacity};ctx->used=0;++ctx->root_epoch;
    if(!count) return 0;
    std::memcpy(ctx->host_positions,positions,count*sizeof(*positions));
    int32_t result=prepare(ctx,error,cap);if(result) return result;
    hipError_t code=hipEventRecord(ctx->transfers[0],ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->positions,ctx->host_positions,count*sizeof(*positions),hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->transfers[1],ctx->stream);
    if(code!=hipSuccess) return transfer_failure(ctx,code,error,cap);
    hipLaunchKernelGGL(generic_legal,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->moves,ctx->counts,ctx->flags,count,&ctx->summary->error);
    code=hipGetLastError();
    if(code==hipSuccess) code=hipEventRecord(ctx->transfers[2],ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->host_moves,ctx->moves,count*CHEN_RULES_MAX_MOVES*sizeof(*moves),hipMemcpyDeviceToHost,ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->host_counts,ctx->counts,count*sizeof(*counts),hipMemcpyDeviceToHost,ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->host_flags,ctx->flags,count*sizeof(*flags),hipMemcpyDeviceToHost,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->transfers[3],ctx->stream);
    if(code!=hipSuccess) return transfer_failure(ctx,code,error,cap);
    result=complete_transfer(ctx,timing,error,cap);if(result) return result;
    std::memcpy(moves,ctx->host_moves,count*CHEN_RULES_MAX_MOVES*sizeof(*moves));
    std::memcpy(counts,ctx->host_counts,count*sizeof(*counts));std::memcpy(flags,ctx->host_flags,count*sizeof(*flags));return 0;
}
extern "C" int32_t chen_rules_make(ChenRulesContext *ctx,const ChenRulePosition *positions,
    const ChenRuleMove *moves,uint64_t count,ChenRulePosition *out,ChenTimings *timing,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !timing || (count&&(!positions||!moves||!out))) return fail(error,cap,"null rules make buffer");
    if(count>ctx->capacity) return fail(error,cap,"rules batch exceeds capacity",CHEN_RULES_CAPACITY);
    if(ctx->root_epoch==UINT64_MAX) return fail(error,cap,"root epoch exhausted; create a new context");
    *timing=ChenTimings{0,0,0,count,ctx->capacity};ctx->used=0;++ctx->root_epoch;
    if(!count) return 0;
    std::memcpy(ctx->host_positions,positions,count*sizeof(*positions));std::memcpy(ctx->host_moves,moves,count*sizeof(*moves));
    int32_t result=prepare(ctx,error,cap);if(result) return result;
    hipError_t code=hipEventRecord(ctx->transfers[0],ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->positions,ctx->host_positions,count*sizeof(*positions),hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->requested,ctx->host_moves,count*sizeof(*moves),hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->transfers[1],ctx->stream);
    if(code!=hipSuccess) return transfer_failure(ctx,code,error,cap);
    hipLaunchKernelGGL(generic_make,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->requested,ctx->moves,ctx->scratch,count,&ctx->summary->error);
    code=hipGetLastError();
    if(code==hipSuccess) code=hipEventRecord(ctx->transfers[2],ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->host_positions,ctx->scratch,count*sizeof(*out),hipMemcpyDeviceToHost,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->transfers[3],ctx->stream);
    if(code!=hipSuccess) return transfer_failure(ctx,code,error,cap);
    result=complete_transfer(ctx,timing,error,cap);if(result) return result;
    std::memcpy(out,ctx->host_positions,count*sizeof(*out));return 0;
}
extern "C" int32_t chen_rules_reset(ChenRulesContext *ctx,const ChenRulePosition *root,
    const uint64_t *history,uint64_t history_count,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !root || (history_count&&!history)) return fail(error,cap,"null rules root/history");
    if(history_count>65535) return fail(error,cap,"root history exceeds 65535 hashes",CHEN_RULES_CAPACITY);
    if(ctx->root_epoch==UINT64_MAX) return fail(error,cap,"root epoch exhausted; create a new context");
    ctx->used=0;ctx->history_count=history_count;ctx->kernel_total_ms=0;ctx->mode=-1;ctx->evaluator=-1;
    ctx->external_callback=nullptr;ctx->external_owner=nullptr;ctx->external_max_batch=0;
    *ctx->host_positions=*root;int32_t result=prepare(ctx,error,cap);if(result) return result;
    hipError_t code=hipMemcpyAsync(ctx->positions,ctx->host_positions,sizeof(*root),hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess && history_count) code=hipMemcpyAsync(ctx->history,history,history_count*sizeof(*history),hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess) code=hipMemsetAsync(ctx->classified,0,sizeof(*ctx->classified),ctx->stream);
    if(code==hipSuccess) code=hipMemsetAsync(ctx->evaluated,0,sizeof(*ctx->evaluated),ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->begin,ctx->stream);
    if(code!=hipSuccess) return transfer_failure(ctx,code,error,cap);
    hipLaunchKernelGGL(reset_root,dim3(1),dim3(1),0,ctx->stream,ctx->positions,ctx->nodes,&ctx->summary->error);
    result=complete(ctx,nullptr,error,cap);if(!result) { ctx->used=1;++ctx->root_epoch; }return result;
}
extern "C" int32_t chen_rules_classify(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    uint32_t adjudicate,ChenRulesStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!stats || !range_valid(ctx,start,count) || adjudicate>1) return fail(error,cap,"invalid classify range/mode");
    int32_t result=lock_mode(ctx,int(adjudicate),error,cap);if(result) return result;
    result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(classify_kernel,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->nodes,ctx->moves,ctx->counts,start,count,adjudicate,ctx->history,ctx->history_count,ctx->classified,&ctx->summary->error);
    double ms=0;result=complete(ctx,&ms,error,cap);return result?result:statistics(ctx,stats,ms,error,cap);
}
extern "C" int32_t chen_rules_expand(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    uint32_t adjudicate,ChenRulesStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!stats || !range_valid(ctx,start,count) || adjudicate>1) return fail(error,cap,"invalid expand range/mode");
    int32_t result=lock_mode(ctx,int(adjudicate),error,cap);if(result) return result;
    result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(expansion_preflight,grid(count),dim3(64),0,ctx->stream,ctx->nodes,start,count,&ctx->summary->error);
    result=complete(ctx,nullptr,error,cap);if(result) return result;
    result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(classify_kernel,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->nodes,ctx->moves,ctx->counts,start,count,adjudicate,ctx->history,ctx->history_count,ctx->classified,&ctx->summary->error);
    double ms=0;result=complete(ctx,&ms,error,cap);if(result) return result;
    // A separate validated prefix phase prevents capacity failures from writing
    // any child ranges or boards. It is deliberately serial onGPU initially.
    result=prepare(ctx,error,cap);if(result) return result;
    hipLaunchKernelGGL(prefix_kernel,dim3(1),dim3(1),0,ctx->stream,ctx->nodes,ctx->counts,start,count,ctx->used,ctx->capacity,ctx->summary);
    double prefix_ms=0;result=complete(ctx,&prefix_ms,error,cap);if(result) return result;
    Summary summary{};hipError_t code=hipMemcpy(&summary,ctx->summary,sizeof(summary),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    const uint64_t next_start=ctx->used,next_count=summary.next_count;
    result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(expand_kernel,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->nodes,ctx->moves,start,count);
    double expansion_ms=0;result=complete(ctx,&expansion_ms,error,cap);if(result) return result;
    ctx->used+=next_count;
    result=statistics(ctx,stats,ms+prefix_ms+expansion_ms,error,cap);
    if(!result) { stats->next_start=next_start;stats->next_count=next_count; }
    return result;
}
extern "C" int32_t chen_rules_evaluate(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    uint32_t evaluator,ChenRulesStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!stats || !range_valid(ctx,start,count) || evaluator>1) return fail(error,cap,"invalid evaluate range/mode");
    if(ctx->evaluator>=0 && ctx->evaluator!=int(evaluator)) return fail(error,cap,"resident evaluator cannot change before reset");
    int32_t result=lock_mode(ctx,1,error,cap);if(result) return result;
    ctx->evaluator=int(evaluator);
    result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(classify_kernel,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->nodes,ctx->moves,ctx->counts,start,count,1,ctx->history,ctx->history_count,ctx->classified,&ctx->summary->error);
    if(count) hipLaunchKernelGGL(evaluate_kernel_rules,grid(count),dim3(64),0,ctx->stream,ctx->positions,ctx->nodes,start,count,evaluator,ctx->evaluated,&ctx->summary->error);
    double ms=0;result=complete(ctx,&ms,error,cap);return result?result:statistics(ctx,stats,ms,error,cap);
}
extern "C" int32_t chen_rules_reduce(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    ChenRulesStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!stats || !range_valid(ctx,start,count)) return fail(error,cap,"invalid reduce range");
    int32_t result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(reduction_preflight,grid(count),dim3(64),0,ctx->stream,ctx->nodes,start,count,ctx->used,&ctx->summary->error);
    result=complete(ctx,nullptr,error,cap);if(result) return result;
    result=prepare(ctx,error,cap);if(result) return result;
    if(count) hipLaunchKernelGGL(reduce_kernel,grid(count),dim3(64),0,ctx->stream,ctx->nodes,start,count,&ctx->summary->error);
    double ms=0;result=complete(ctx,&ms,error,cap);return result?result:statistics(ctx,stats,ms,error,cap);
}
extern "C" int32_t chen_rules_finish(ChenRulesContext *ctx,ChenRulesResult *out,ChenRuleMove *pv,
    uint64_t pv_capacity,ChenRulesStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !ctx->used || !out || !stats || (pv_capacity&&!pv)) return fail(error,cap,"invalid finish buffers/tree");
    int32_t result=prepare(ctx,error,cap);if(result) return result;
    hipLaunchKernelGGL(finish_kernel,dim3(1),dim3(1),0,ctx->stream,ctx->nodes,ctx->result,ctx->requested,ctx->used,&ctx->summary->error);
    double ms=0;result=complete(ctx,&ms,error,cap);if(result) return result;
    ChenRulesResult root{};hipError_t code=hipMemcpy(&root,ctx->result,sizeof(root),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    if(root.pv_length>pv_capacity) return fail(error,cap,"PV output capacity insufficient",CHEN_RULES_CAPACITY);
    if(root.pv_length) code=hipMemcpy(pv,ctx->requested,root.pv_length*sizeof(*pv),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    *out=root;return statistics(ctx,stats,ms,error,cap);
}
extern "C" int32_t chen_rules_download(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    ChenRulePosition *out,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!range_valid(ctx,start,count) || (count&&!out)) return fail(error,cap,"invalid download range/buffer");
    hipError_t code=hipSetDevice(ctx->device);
    if(code==hipSuccess && count) code=hipMemcpy(out,ctx->positions+start,count*sizeof(*out),hipMemcpyDeviceToHost);
    return code==hipSuccess?0:hip_fail(error,cap,code);
}
extern "C" int32_t chen_rules_perft(ChenRulesContext *ctx,const ChenRulePosition *root,uint32_t depth,
    uint64_t *nodes,ChenRulesStats *stats,char *error,uint64_t cap) {
    if(!nodes || !stats || depth>64) return fail(error,cap,"perft needs outputs and depth<=64");
    int32_t result=chen_rules_reset(ctx,root,nullptr,0,error,cap);if(result) return result;
    uint64_t start=0,count=1;
    for(uint32_t d=0;d<depth && count;++d) {
        result=chen_rules_expand(ctx,start,count,0,stats,error,cap);if(result) return result;
        start=stats->next_start;count=stats->next_count;
    }
    *nodes=count;result=statistics(ctx,stats,0,error,cap);
    if(!result) { stats->next_start=start;stats->next_count=count; }return result;
}

// Replay export ABI is independent of rules ABI; mutable arena pointers stay private.
namespace {
__device__ int64_t replay_action(ChenRuleMove move,uint32_t side) {
    const uint32_t flip=side?56:0;
    return int64_t(((uint32_t(move.from)^flip)*64+(uint32_t(move.to)^flip))*5+move.promotion);
}
__global__ void replay_preflight(const ChenRulePosition *positions,const Node *nodes,
    const ChenRuleMove *moves,const uint32_t *counts,uint32_t *terminals,
    uint64_t start,uint64_t count,uint64_t used,uint32_t slots,
    const uint64_t *history,uint64_t history_count,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t child_id=start+row;
    const Node child=nodes[child_id];
    if(child.parent>=used) { device_error(error,4);return; }
    const Node parent=nodes[child.parent];
    if(!(parent.flags&EXPANDED) || (parent.flags&TERMINAL) ||
        !parent.child_count || parent.first_child>child_id ||
        child_id-parent.first_child>=parent.child_count ||
        parent.child_count!=counts[child.parent]) { device_error(error,4);return; }
    if(parent.child_count>slots) { device_error(error,2);return; }
    const ChenRuleMove expected=moves[child.parent*CHEN_RULES_MAX_MOVES+child_id-parent.first_child];
    if(child.move.from!=expected.from || child.move.to!=expected.to ||
        child.move.promotion!=expected.promotion || child.move.reserved) {
        device_error(error,4);return;
    }
    if(child.flags&CLASSIFIED) terminals[row]=(child.flags&TERMINAL)?1:0;
    else {
        const ChenRulePosition p=positions[child_id];
        const uint32_t legal=legal_moves(p,nullptr,error);
        terminals[row]=!legal || p.halfmove>=100 || insufficient(p) ||
            repeated(positions,nodes,child_id,history,history_count);
    }
}
__device__ void replay_state(const ChenRulePosition &p,uint64_t boards_address,
    uint64_t state_address,uint64_t row) {
    auto *boards=reinterpret_cast<uint64_t *>(boards_address)+row*12;
    auto *state=reinterpret_cast<uint32_t *>(state_address)+row*5;
    for(int i=0;i<12;++i) boards[i]=p.boards[i];
    state[0]=p.side;state[1]=p.castling;state[2]=uint32_t(p.ep_square);
    state[3]=p.halfmove;state[4]=p.fullmove;
}
__global__ void replay_export_kernel(const ChenRulePosition *positions,const Node *nodes,
    const ChenRuleMove *moves,const uint32_t *terminals,uint64_t start,uint64_t count,
    uint64_t epoch,ChenReplaySink sink) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t child_id=start+row,parent_id=nodes[child_id].parent;
    const ChenRulePosition parent=positions[parent_id],child=positions[child_id];
    replay_state(parent,sink.pieces,sink.state,row);
    replay_state(child,sink.next_pieces,sink.next_state,row);
    auto *legal=reinterpret_cast<int16_t *>(sink.legal_actions)+row*sink.legal_slots;
    const uint32_t n=nodes[parent_id].child_count;
    for(uint32_t i=0;i<sink.legal_slots;++i) legal[i]=-1;
    // Bounded insertion sort avoids a dense 20480-action support allocation.
    for(uint32_t i=0;i<n;++i) {
        const int16_t action=int16_t(replay_action(moves[parent_id*CHEN_RULES_MAX_MOVES+i],parent.side));
        uint32_t j=i;
        while(j && legal[j-1]>action) { legal[j]=legal[j-1];--j; }
        legal[j]=action;
    }
    reinterpret_cast<int32_t *>(sink.legal_counts)[row]=int32_t(n);
    reinterpret_cast<int64_t *>(sink.actions)[row]=replay_action(nodes[child_id].move,parent.side);
    reinterpret_cast<float *>(sink.material_proxy)[row]=tanhf(float(evaluate(parent,0))/800.0f);
    reinterpret_cast<uint8_t *>(sink.terminal)[row]=uint8_t(terminals[row]);
    reinterpret_cast<uint64_t *>(sink.parent_hash)[row]=parent.hash;
    reinterpret_cast<uint64_t *>(sink.child_hash)[row]=child.hash;
    reinterpret_cast<uint64_t *>(sink.parent_node)[row]=parent_id;
    reinterpret_cast<uint64_t *>(sink.child_node)[row]=child_id;
    reinterpret_cast<uint64_t *>(sink.root_epoch)[row]=epoch;
}
struct ReplayRange { uint64_t address,bytes; };
bool overlaps(ReplayRange a,ReplayRange b) {
    return a.address<b.address+b.bytes && b.address<a.address+a.bytes;
}
int32_t export_intervals_valid(ChenRulesContext *ctx,const uint64_t *addresses,
    const uint64_t *strides,const uint64_t *alignments,uint32_t n,uint64_t capacity,
    char *error,uint64_t cap) {
    if(n>32) return fail(error,cap,"too many typed export intervals");
    ReplayRange ranges[32]{};
    for(uint32_t i=0;i<n;++i) {
        const uint64_t bytes=capacity*strides[i],address=addresses[i];
        if(!bytes) continue;
        if(!address || address%alignments[i] || address>UINT64_MAX-bytes)
            return fail(error,cap,"null, misaligned or overflowing replay sink interval");
        hipPointerAttribute_t attributes{};
        hipError_t code=hipPointerGetAttributes(&attributes,reinterpret_cast<void *>(address));
        if(code!=hipSuccess) return hip_fail(error,cap,code);
        if(attributes.type!=hipMemoryTypeDevice || attributes.isManaged || attributes.device!=ctx->device)
            return fail(error,cap,"replay sink requires device allocations on the context device");
        hipDeviceptr_t base=nullptr;size_t allocation=0;
        code=hipMemGetAddressRange(&base,&allocation,reinterpret_cast<hipDeviceptr_t>(address));
        if(code!=hipSuccess) return hip_fail(error,cap,code);
        const uint64_t beginning=reinterpret_cast<uint64_t>(base);
        if(address<beginning || address-beginning>allocation || bytes>allocation-(address-beginning))
            return fail(error,cap,"replay sink interval exceeds GPU allocation bounds");
        ranges[i]={address,bytes};
        for(uint32_t j=0;j<i;++j) if(ranges[j].bytes && overlaps(ranges[i],ranges[j]))
            return fail(error,cap,"replay sink output intervals overlap");
    }
    // Even guessed internal pointers cannot turn the snapshot into a live alias.
    const ReplayRange arena[]={
        {reinterpret_cast<uint64_t>(ctx->positions),ctx->capacity*sizeof(*ctx->positions)},
        {reinterpret_cast<uint64_t>(ctx->scratch),ctx->capacity*sizeof(*ctx->scratch)},
        {reinterpret_cast<uint64_t>(ctx->moves),ctx->capacity*CHEN_RULES_MAX_MOVES*sizeof(*ctx->moves)},
        {reinterpret_cast<uint64_t>(ctx->requested),ctx->capacity*sizeof(*ctx->requested)},
        {reinterpret_cast<uint64_t>(ctx->counts),ctx->capacity*sizeof(*ctx->counts)},
        {reinterpret_cast<uint64_t>(ctx->flags),ctx->capacity*sizeof(*ctx->flags)},
        {reinterpret_cast<uint64_t>(ctx->nodes),ctx->capacity*sizeof(*ctx->nodes)},
        {reinterpret_cast<uint64_t>(ctx->history),65535*sizeof(*ctx->history)},
        {reinterpret_cast<uint64_t>(ctx->classified),sizeof(*ctx->classified)},
        {reinterpret_cast<uint64_t>(ctx->evaluated),sizeof(*ctx->evaluated)},
        {reinterpret_cast<uint64_t>(ctx->summary),sizeof(*ctx->summary)},
        {reinterpret_cast<uint64_t>(ctx->result),sizeof(*ctx->result)},
        {reinterpret_cast<uint64_t>(ctx->extra_private[0]),ctx->extra_private_bytes[0]},
        {reinterpret_cast<uint64_t>(ctx->extra_private[1]),ctx->extra_private_bytes[1]},
        {reinterpret_cast<uint64_t>(ctx->extra_private[2]),ctx->extra_private_bytes[2]},
        {reinterpret_cast<uint64_t>(ctx->extra_private[3]),ctx->extra_private_bytes[3]}};
    for(const auto range:ranges) if(range.bytes)
        for(const auto internal:arena) if(overlaps(range,internal))
            return fail(error,cap,"replay sink aliases private arena memory");
    return 0;
}
int32_t replay_sink_valid(ChenRulesContext *ctx,const ChenReplaySink &sink,
    char *error,uint64_t cap) {
    if(sink.abi_version!=1 || sink.struct_size!=sizeof(ChenReplaySink) ||
        sink.device!=ctx->device || !sink.legal_slots || sink.legal_slots>CHEN_RULES_MAX_MOVES ||
        sink.capacity>uint64_t(INT32_MAX)) return fail(error,cap,"invalid replay sink schema/device/capacity");
    const uint64_t addresses[]={sink.pieces,sink.state,sink.next_pieces,sink.next_state,
        sink.legal_actions,sink.legal_counts,sink.actions,sink.material_proxy,sink.terminal,
        sink.parent_hash,sink.child_hash,sink.parent_node,sink.child_node,sink.root_epoch};
    const uint64_t strides[]={96,20,96,20,uint64_t(sink.legal_slots)*2,4,8,4,1,8,8,8,8,8};
    const uint64_t alignments[]={8,4,8,4,2,4,8,4,1,8,8,8,8,8,8};
    return export_intervals_valid(ctx,addresses,strides,alignments,14,sink.capacity,error,cap);
}
}
extern "C" uint32_t chen_replay_abi_version() { return 1; }
extern "C" uint64_t chen_replay_runtime_identity(uint32_t selector) {
    switch(selector) {
        case 0:return reinterpret_cast<uint64_t>(&hipPointerGetAttributes);
        case 1:return reinterpret_cast<uint64_t>(&hipMemGetAddressRange);
        case 2:return reinterpret_cast<uint64_t>(static_cast<hipError_t (*)(void **,size_t)>(&hipMalloc));
        case 3:return reinterpret_cast<uint64_t>(&hipFree);
        case 4:return reinterpret_cast<uint64_t>(&hipStreamSynchronize);
        default:return 0;
    }
}
extern "C" int32_t chen_replay_epoch(ChenRulesContext *ctx,uint64_t *epoch,
    char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !epoch) return fail(error,cap,"null replay epoch context/output");
    *epoch=ctx->root_epoch;return 0;
}
extern "C" int32_t chen_rules_export_transitions(ChenRulesContext *ctx,uint64_t child_start,
    uint64_t count,uint64_t expected_epoch,const ChenReplaySink *sink,ChenReplayExport *out,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(ctx && expected_epoch!=ctx->root_epoch) return fail(error,cap,"stale replay root epoch");
    if(!ctx || !sink || !out || !range_valid(ctx,child_start,count) ||
        (count && !child_start) || ctx->mode!=1) return fail(error,cap,"invalid replay selection or non-search arena");
    hipError_t code=hipSetDevice(ctx->device);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    int32_t status=replay_sink_valid(ctx,*sink,error,cap);if(status) return status;
    if(count>sink->capacity) return fail(error,cap,"replay sink row capacity exceeded",CHEN_RULES_CAPACITY);
    if(!count) { *out=ChenReplayExport{0,ctx->root_epoch,0};return 0; }
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(replay_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,ctx->moves,ctx->counts,ctx->flags,child_start,count,
        ctx->used,sink->legal_slots,ctx->history,ctx->history_count,&ctx->summary->error);
    double validation_ms=0;status=complete(ctx,&validation_ms,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(replay_export_kernel,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,ctx->moves,ctx->flags,child_start,count,ctx->root_epoch,*sink);
    double export_ms=0;status=complete(ctx,&export_ms,error,cap);if(status) return status;
    *out=ChenReplayExport{count,ctx->root_epoch,validation_ms+export_ms};return 0;
}

namespace {
__global__ void teacher_preflight(const Node *nodes,const ChenRuleMove *moves,
    const uint32_t *counts,uint64_t start,uint64_t count,uint64_t used,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t child_id=start+row;
    const Node edge=nodes[child_id];
    if(edge.parent>=used) { device_error(error,4);return; }
    const Node parent=nodes[edge.parent];
    if(!(parent.flags&EXPANDED) || !(parent.flags&SCORED) || (parent.flags&TERMINAL) ||
        !parent.child_count || parent.first_child>=used || parent.child_count>used-parent.first_child ||
        child_id<parent.first_child || child_id-parent.first_child>=parent.child_count ||
        parent.child_count!=counts[edge.parent]) { device_error(error,4);return; }
    const ChenRuleMove expected=moves[edge.parent*CHEN_RULES_MAX_MOVES+child_id-parent.first_child];
    if(edge.move.from!=expected.from || edge.move.to!=expected.to ||
        edge.move.promotion!=expected.promotion || edge.move.reserved) { device_error(error,4);return; }
    int best=-MATE-1;uint64_t winner=NONE;
    for(uint32_t i=0;i<parent.child_count;++i) {
        const uint64_t child=parent.first_child+i;
        if(!(nodes[child].flags&SCORED)) { device_error(error,4);return; }
        const int score=-nodes[child].score;
        if(score>best) { best=score;winner=child; }
    }
    if(parent.best!=winner || parent.score!=best) device_error(error,4);
}
__global__ void teacher_export_kernel(const ChenRulePosition *positions,const Node *nodes,
    uint64_t start,uint64_t count,ChenTeacherSink sink) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t parent_id=nodes[start+row].parent;
    const Node parent=nodes[parent_id];
    reinterpret_cast<int32_t *>(sink.scores)[row]=parent.score;
    reinterpret_cast<int64_t *>(sink.best_actions)[row]=replay_action(nodes[parent.best].move,positions[parent_id].side);
    reinterpret_cast<int32_t *>(sink.parent_ply)[row]=int32_t(parent.ply);
}
}
extern "C" uint32_t chen_teacher_abi_version() { return 1; }
extern "C" int32_t chen_rules_export_teacher(ChenRulesContext *ctx,uint64_t child_start,
    uint64_t count,uint64_t expected_epoch,const ChenTeacherSink *sink,ChenTeacherExport *out,
    char *error,uint64_t cap) {
    clear_error(error,cap);
    if(ctx && expected_epoch!=ctx->root_epoch) return fail(error,cap,"stale teacher root epoch");
    if(!ctx || !sink || !out || !range_valid(ctx,child_start,count) || (count && !child_start) || ctx->mode!=1 || ctx->evaluator>1)
        return fail(error,cap,"invalid teacher selection or non-search arena");
    if(sink->abi_version!=1 || sink->struct_size!=sizeof(ChenTeacherSink) || sink->device!=ctx->device ||
        sink->reserved || sink->capacity>uint64_t(INT32_MAX)) return fail(error,cap,"invalid teacher sink schema/device/capacity");
    hipError_t code=hipSetDevice(ctx->device);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    const uint64_t addresses[]={sink->scores,sink->best_actions,sink->parent_ply};
    const uint64_t strides[]={4,8,4},alignments[]={4,8,4};
    int32_t status=export_intervals_valid(ctx,addresses,strides,alignments,3,sink->capacity,error,cap);
    if(status) return status;
    if(count>sink->capacity) return fail(error,cap,"teacher sink row capacity exceeded",CHEN_RULES_CAPACITY);
    if(!count) { *out=ChenTeacherExport{0,ctx->root_epoch,0,ctx->evaluator,1};return 0; }
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(teacher_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->nodes,ctx->moves,ctx->counts,child_start,count,ctx->used,&ctx->summary->error);
    double validation_ms=0;status=complete(ctx,&validation_ms,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(teacher_export_kernel,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,child_start,count,*sink);
    double export_ms=0;status=complete(ctx,&export_ms,error,cap);if(status) return status;
    *out=ChenTeacherExport{count,ctx->root_epoch,validation_ms+export_ms,ctx->evaluator,1};return 0;
}

namespace {
__global__ void external_phase_preflight(const Node *nodes,uint64_t start,uint64_t count,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row<count && (nodes[start+row].flags&EXPANDED)) device_error(error,4);
}
__global__ void external_scores_initialize(const Node *nodes,uint32_t *scratch,uint64_t start,
    uint64_t count,Summary *summary) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t i=start+row;
    scratch[i]=uint32_t(INT32_MIN);
    if(!(nodes[i].flags&SCORED)) atomicAdd(reinterpret_cast<unsigned long long *>(&summary->next_count),1ull);
}
__global__ void external_score_preflight(const Node *nodes,const uint32_t *scratch,uint64_t start,
    uint64_t count,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t i=start+row;
    if(!(nodes[i].flags&SCORED)) {
        const int32_t score=int32_t(scratch[i]);
        if(score<-MAX_EVAL || score>MAX_EVAL) device_error(error,5);
    }
}
__global__ void external_score_commit(Node *nodes,const uint32_t *scratch,uint64_t start,
    uint64_t count,unsigned long long *evaluated) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t i=start+row;
    if(!(nodes[i].flags&SCORED)) {
        nodes[i].score=int32_t(scratch[i]);nodes[i].flags|=SCORED;
        atomicAdd(evaluated,1ull);
    }
}
}
extern "C" uint32_t chen_external_abi_version() { return 1; }
extern "C" int32_t chen_rules_evaluate_external(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    const ChenExternalEvaluator *desc,ChenRulesStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !desc || !stats || !range_valid(ctx,start,count))
        return fail(error,cap,"invalid external evaluation range/buffers");
    if(desc->abi_version!=1 || desc->struct_size!=sizeof(ChenExternalEvaluator) ||
        !desc->evaluate || !desc->max_batch || desc->max_batch>uint64_t(INT32_MAX) || desc->reserved)
        return fail(error,cap,"invalid external evaluator descriptor");
    for(uint32_t i=0;i<5;++i)
        if(desc->runtime_identity[i]!=chen_replay_runtime_identity(i))
            return fail(error,cap,"external evaluator HIP runtime identity mismatch");
    if(ctx->evaluator>=0 && ctx->evaluator!=2)
        return fail(error,cap,"resident evaluator cannot change before reset");
    if(ctx->external_callback && (ctx->external_callback!=desc->evaluate || ctx->external_owner!=desc->owner ||
        ctx->external_max_batch!=desc->max_batch)) return fail(error,cap,"external evaluator identity cannot change before reset");
    int32_t status=lock_mode(ctx,1,error,cap);if(status) return status;
    ctx->evaluator=2;ctx->external_callback=desc->evaluate;ctx->external_owner=desc->owner;ctx->external_max_batch=desc->max_batch;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    if(count) hipLaunchKernelGGL(external_phase_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->nodes,start,count,&ctx->summary->error);
    status=complete(ctx,nullptr,error,cap);if(status) return status;
    // Terminal adjudication remains owned by rules. It may commit exact terminal
    // scores even if the later external callback fails; no static scores commit.
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    if(count) hipLaunchKernelGGL(classify_kernel,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,ctx->moves,ctx->counts,start,count,1,
        ctx->history,ctx->history_count,ctx->classified,&ctx->summary->error);
    status=complete(ctx,nullptr,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    if(count) hipLaunchKernelGGL(external_scores_initialize,grid(count),dim3(64),0,ctx->stream,
        ctx->nodes,ctx->flags,start,count,ctx->summary);
    status=complete(ctx,nullptr,error,cap);if(status) return status;
    Summary pending{};
    hipError_t code=hipMemcpy(&pending,ctx->summary,sizeof(pending),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    if(!pending.next_count) return statistics(ctx,stats,0,error,cap);
    for(uint64_t offset=0;offset<count;) {
        const uint64_t n=(count-offset<desc->max_batch)?count-offset:desc->max_batch;
        code=hipSetDevice(ctx->device);
        if(code==hipSuccess) code=hipEventRecord(ctx->begin,ctx->stream);
        if(code==hipSuccess) code=hipStreamSynchronize(ctx->stream);
        if(code!=hipSuccess) { (void)hipStreamSynchronize(ctx->stream);return hip_fail(error,cap,code); }
        int32_t callback_status=0;
        try {
            callback_status=desc->evaluate(desc->owner,ctx->device,ctx->positions+start+offset,n,
                reinterpret_cast<int32_t *>(ctx->flags+start+offset),error,cap);
        } catch(...) {
            fail(error,cap,"external callback crossed C ABI with an exception");
            callback_status=-1;
        }
        code=hipSetDevice(ctx->device);
        if(code==hipSuccess) code=hipEventRecord(ctx->end,ctx->stream);
        if(code==hipSuccess) code=hipStreamSynchronize(ctx->stream);
        float callback_ms=0;
        if(code==hipSuccess) code=hipEventElapsedTime(&callback_ms,ctx->begin,ctx->end);
        if(code==hipSuccess) ctx->kernel_total_ms+=callback_ms;
        if(callback_status) {
            if(error && cap && !*error) fail(error,cap,"external evaluator callback failed");
            return -1;
        }
        if(code!=hipSuccess) return hip_fail(error,cap,code);
        offset+=n;
    }
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(external_score_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->nodes,ctx->flags,start,count,&ctx->summary->error);
    status=complete(ctx,nullptr,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(external_score_commit,grid(count),dim3(64),0,ctx->stream,
        ctx->nodes,ctx->flags,start,count,ctx->evaluated);
    status=complete(ctx,nullptr,error,cap);return status?status:statistics(ctx,stats,0,error,cap);
}

// Independently versioned selective arena; full-width ABI1 above stays intact.
#include "chen_selective.inc"
#include "chen_symbolic.inc"
#include "chen_tactical.inc"
