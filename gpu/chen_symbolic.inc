// Shares exact private position priors and resident ownership checks.
#include "chen_symbolic.h"
namespace {
__device__ uint64_t symbolic_pinned(const ChenRulePosition &p,int color) {
    const int king=__ffsll(static_cast<unsigned long long>(p.boards[color*6+5]))-1;
    uint64_t result=0;
    for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) {
        if(!dx && !dy) continue;
        int candidate=-1,x=king%8+dx,y=king/8+dy;
        while(x>=0 && x<8 && y>=0 && y<8) {
            const int square=y*8+x,piece=piece_at(p,square);
            if(piece>=0) {
                if(candidate<0) {
                    if(piece/6!=color) break;
                    candidate=square;
                } else {
                    const int kind=piece%6;
                    if(piece/6!=color && (kind==4 || kind==(dx && dy?2:3)))
                        result |= uint64_t(1)<<candidate;
                    break;
                }
            }
            x+=dx;y+=dy;
        }
    }
    return result;
}
__device__ uint64_t symbolic_outgoing(const ChenRulePosition &p,int square,int color,uint64_t occ) {
    const int piece=piece_at(p,square);
    if(piece<0 || piece/6!=color) return 0;
    const int kind=piece%6;
    return kind==0 || kind==1 || kind==5?leaper(square,kind,color):
        sliding(square,occ,kind==2 || kind==4,kind==3 || kind==4);
}
__global__ void symbolic_preflight(const ChenRulePosition *positions,const Node *nodes,
    uint64_t start,uint64_t count,uint64_t used,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t source=start+row;
    if(!valid_position(positions[source])) { device_error(error,1);return; }
    uint64_t index=source;
    while(index) {
        const Node child=nodes[index];
        if(child.parent>=index || child.reserved || child.move.reserved || !child.ply) {
            device_error(error,4);return;
        }
        const Node parent=nodes[child.parent];
        if(!(parent.flags&EXPANDED) || (parent.flags&TERMINAL) ||
            parent.ply+1!=child.ply || index<parent.first_child ||
            index-parent.first_child>=parent.child_count || parent.first_child>=used ||
            parent.child_count>used-parent.first_child) { device_error(error,4);return; }
        index=child.parent;
    }
    if(nodes[0].parent!=NONE || nodes[0].ply || nodes[0].reserved) device_error(error,4);
}
__global__ void symbolic_export_kernel(const ChenRulePosition *positions,const Node *nodes,
    const uint64_t *history,uint64_t history_count,uint64_t start,uint64_t count,
    uint64_t epoch,ChenSymbolicSink sink) {
    const uint64_t row=blockIdx.x;
    if(row>=count) return;
    const int square=int(threadIdx.x);
    const uint64_t index=start+row;
    const ChenRulePosition p=positions[index];
    __shared__ uint64_t pins[2],checkers,occ;
    __shared__ int64_t counters[4];
    __shared__ float phase;
    if(!square) {
        pins[0]=symbolic_pinned(p,int(p.side));pins[1]=symbolic_pinned(p,1-int(p.side));
        const int king=__ffsll(static_cast<unsigned long long>(p.boards[p.side*6+5]))-1;
        checkers=attackers(p,king,1-int(p.side));occ=occupied(p);
        constexpr int weights[6]={0,1,1,2,4,0};int value=0;
        for(int piece=0;piece<12;++piece) value+=weights[piece%6]*__popcll(p.boards[piece]);
        phase=float(value>24?24:value)/24.0f;
        counters[0]=p.halfmove;counters[1]=0;counters[2]=0;counters[3]=nodes[index].ply;
        uint32_t remaining=p.halfmove+1;uint64_t ancestor=index;
        while(remaining) {
            counters[1]+=positions[ancestor].hash==p.hash;++counters[2];--remaining;
            if(nodes[ancestor].parent==NONE) break;
            ancestor=nodes[ancestor].parent;
        }
        uint64_t past=history_count;
        while(remaining && past) {
            counters[1]+=history[--past]==p.hash;++counters[2];--remaining;
        }
        for(int h=0;h<4;++h) reinterpret_cast<int64_t *>(sink.history)[row*4+h]=counters[h];
        reinterpret_cast<uint64_t *>(sink.position_hash)[row]=p.hash;
        reinterpret_cast<uint64_t *>(sink.node_id)[row]=index;
        reinterpret_cast<uint64_t *>(sink.root_epoch)[row]=epoch;
    }
    __syncthreads();
    const int raw=square^(p.side?56:0);
    auto *features=reinterpret_cast<float *>(sink.features)+row*18*64;
    for(int piece=0;piece<12;++piece) {
        const int absolute=((piece/6)^int(p.side))*6+piece%6;
        features[piece*64+square]=float((p.boards[absolute]>>raw)&1);
    }
    for(int right=0;right<4;++right) {
        const int absolute=(((right/2)^int(p.side))*2)+right%2;
        features[(12+right)*64+square]=float((p.castling>>absolute)&1);
    }
    features[16*64+square]=float(raw==p.ep_square);
    features[17*64+square]=float(p.halfmove>100?100:p.halfmove)/100.0f;
    auto *channels=reinterpret_cast<float *>(sink.channels)+(row*64+square)*CHEN_SYMBOLIC_CHANNELS;
    for(int color=0;color<2;++color) {
        const int absolute=color^int(p.side);
        const int incoming=__popcll(attackers(p,raw,absolute));
        const int outgoing=__popcll(symbolic_outgoing(p,raw,absolute,occ));
        channels[color]=float(incoming>8?8:incoming)/8.0f;
        channels[2+color]=float(outgoing>8?8:outgoing)/8.0f;
        channels[4+color]=float((pins[color]>>raw)&1);
    }
    channels[6]=float((checkers>>raw)&1);channels[7]=phase;
    constexpr float scales[4]={100,3,256,256};
    for(int h=0;h<4;++h) channels[8+h]=fminf(float(counters[h])/scales[h],1.0f);
}
__device__ uint64_t topology_orient(uint64_t mask,uint32_t side) {
    // Reversing rank bytes maps every target square s to s XOR56.
    return side?__builtin_bswap64(mask):mask;
}
__device__ int topology_pinner(const ChenRulePosition &p,int square,int color) {
    // Called only for a bit admitted by symbolic_pinned. Continue its king ray
    // past the pinned piece to identify the same first enemy slider.
    const int king=__ffsll(static_cast<unsigned long long>(p.boards[color*6+5]))-1;
    const int dx=(square%8>king%8)-(square%8<king%8);
    const int dy=(square/8>king/8)-(square/8<king/8);
    int x=square%8+dx,y=square/8+dy;
    while(x>=0 && x<8 && y>=0 && y<8) {
        const int candidate=y*8+x,piece=piece_at(p,candidate);
        if(piece>=0) return candidate;
        x+=dx;y+=dy;
    }
    return -1;
}
__global__ void topology_export_kernel(const ChenRulePosition *positions,uint64_t start,
    uint64_t count,uint64_t epoch,ChenTopologySink sink) {
    const uint64_t row=blockIdx.x;
    if(row>=count) return;
    const int source=int(threadIdx.x);
    const uint64_t index=start+row;
    const ChenRulePosition p=positions[index];
    __shared__ uint64_t occ,raw_pins[2];
    if(!source) {
        occ=occupied(p);
        raw_pins[0]=symbolic_pinned(p,int(p.side));
        raw_pins[1]=symbolic_pinned(p,1-int(p.side));
        auto *pins=reinterpret_cast<uint64_t *>(sink.pinned)+row*2;
        pins[0]=topology_orient(raw_pins[0],p.side);
        pins[1]=topology_orient(raw_pins[1],p.side);
        const int king=__ffsll(static_cast<unsigned long long>(p.boards[p.side*6+5]))-1;
        reinterpret_cast<uint64_t *>(sink.checkers)[row]=
            topology_orient(attackers(p,king,1-int(p.side)),p.side);
        reinterpret_cast<uint64_t *>(sink.position_hash)[row]=p.hash;
        reinterpret_cast<uint64_t *>(sink.node_id)[row]=index;
        reinterpret_cast<uint64_t *>(sink.root_epoch)[row]=epoch;
    }
    __syncthreads();
    const int raw=source^(p.side?56:0),piece=piece_at(p,raw);
    const uint64_t attacks=piece<0?0:symbolic_outgoing(p,raw,piece/6,occ);
    reinterpret_cast<uint64_t *>(sink.source_attacks)[row*64+source]=topology_orient(attacks,p.side);
    const int color=piece<0?0:piece/6;
    const bool pinned=piece>=0 && ((raw_pins[color^int(p.side)]>>raw)&1);
    const int pinner=pinned?topology_pinner(p,raw,color):-1;
    reinterpret_cast<int64_t *>(sink.pinner_by_square)[row*64+source]=
        pinner<0?-1:int64_t(pinner^(p.side?56:0));
}
}
extern "C" uint32_t chen_symbolic_abi_version() { return 1; }
extern "C" int32_t chen_rules_export_symbolic(ChenRulesContext *ctx,uint64_t start,
    uint64_t count,uint64_t epoch,const ChenSymbolicSink *sink,ChenSymbolicExport *out,
    char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !sink || !out || !range_valid(ctx,start,count) || ctx->mode==0)
        return fail(error,cap,"invalid symbolic range or non-search arena");
    if(epoch!=ctx->root_epoch) return fail(error,cap,"stale symbolic root epoch");
    if(sink->abi_version!=1 || sink->struct_size!=sizeof(*sink) || sink->reserved ||
        sink->device!=ctx->device || sink->capacity>uint64_t(INT32_MAX))
        return fail(error,cap,"invalid symbolic schema/device/capacity");
    hipError_t code=hipSetDevice(ctx->device);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    const uint64_t addresses[]={sink->features,sink->channels,sink->history,
        sink->position_hash,sink->node_id,sink->root_epoch};
    const uint64_t strides[]={18*64*4,64*CHEN_SYMBOLIC_CHANNELS*4,4*8,8,8,8};
    const uint64_t alignments[]={4,4,8,8,8,8};
    int32_t status=export_intervals_valid(ctx,addresses,strides,alignments,6,sink->capacity,error,cap);
    if(status) return status;
    if(count>sink->capacity) return fail(error,cap,"symbolic sink capacity exceeded",CHEN_RULES_CAPACITY);
    if(!count) { *out=ChenSymbolicExport{0,epoch,0};return 0; }
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(symbolic_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,start,count,ctx->used,&ctx->summary->error);
    double preflight_ms=0;status=complete(ctx,&preflight_ms,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(symbolic_export_kernel,dim3(uint32_t(count)),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,ctx->history,ctx->history_count,start,count,epoch,*sink);
    double export_ms=0;status=complete(ctx,&export_ms,error,cap);if(status) return status;
    *out=ChenSymbolicExport{count,epoch,preflight_ms+export_ms};return 0;
}
extern "C" uint32_t chen_topology_abi_version() { return 1; }
extern "C" int32_t chen_rules_export_topology(ChenRulesContext *ctx,uint64_t start,
    uint64_t count,uint64_t epoch,const ChenTopologySink *sink,ChenSymbolicExport *out,
    char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !sink || !out || !range_valid(ctx,start,count) || ctx->mode==0)
        return fail(error,cap,"invalid topology range or non-search arena");
    if(epoch!=ctx->root_epoch) return fail(error,cap,"stale topology root epoch");
    if(sink->abi_version!=1 || sink->struct_size!=sizeof(*sink) || sink->reserved ||
        sink->device!=ctx->device || sink->capacity>uint64_t(INT32_MAX))
        return fail(error,cap,"invalid topology schema/device/capacity");
    hipError_t code=hipSetDevice(ctx->device);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    const uint64_t addresses[]={sink->source_attacks,sink->pinned,sink->checkers,sink->pinner_by_square,
        sink->position_hash,sink->node_id,sink->root_epoch};
    const uint64_t strides[]={64*8,2*8,8,64*8,8,8,8};
    const uint64_t alignments[]={8,8,8,8,8,8,8};
    int32_t status=export_intervals_valid(ctx,addresses,strides,alignments,7,sink->capacity,error,cap);
    if(status) return status;
    if(count>sink->capacity) return fail(error,cap,"topology sink capacity exceeded",CHEN_RULES_CAPACITY);
    if(!count) { *out=ChenSymbolicExport{0,epoch,0};return 0; }
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(symbolic_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,start,count,ctx->used,&ctx->summary->error);
    double preflight_ms=0;status=complete(ctx,&preflight_ms,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(topology_export_kernel,dim3(uint32_t(count)),dim3(64),0,ctx->stream,
        ctx->positions,start,count,epoch,*sink);
    double export_ms=0;status=complete(ctx,&export_ms,error,cap);if(status) return status;
    *out=ChenSymbolicExport{count,epoch,preflight_ms+export_ms};return 0;
}
