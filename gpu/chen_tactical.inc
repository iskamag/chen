#include "chen_tactical.h"
namespace {
constexpr int TP=32,TK=256,TS=64;
__device__ int tactical_code(const ChenRulePosition &p,int canonical) {
    const int piece=piece_at(p,canonical^(p.side?56:0));
    return piece<0?0:piece%6+1+((piece/6)^int(p.side))*6;
}
__device__ void tactical_history(const ChenRulePosition *positions,const Node *nodes,
    const uint64_t *history,uint64_t count,uint64_t index,int64_t *out) {
    const auto p=positions[index];
    out[0]=p.halfmove;out[1]=0;out[2]=0;out[3]=nodes[index].ply;
    uint32_t remaining=p.halfmove+1;
    uint64_t ancestor=index;
    while(remaining) {
        out[1]+=positions[ancestor].hash==p.hash;++out[2];--remaining;
        if(nodes[ancestor].parent==NONE) break;
        ancestor=nodes[ancestor].parent;
    }
    while(remaining && count) {
        out[1]+=history[--count]==p.hash;++out[2];--remaining;
    }
}
__global__ void tactical_piece_preflight(const ChenRulePosition *positions,uint64_t start,
    uint64_t count,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row<count && __popcll(occupied(positions[start+row]))>TP) device_error(error,2);
}
struct TacticalWriter {
    ChenTacticalSink sink;uint64_t row;uint32_t edges;uint32_t *error;
    __device__ void add(int source,int receiver,int kind,int ss,int rs,int old,int next) {
        if(edges>=sink.edge_slots) { device_error(error,2);return; }
        const uint64_t at=row*sink.edge_slots+edges++;
        auto *indices=reinterpret_cast<int64_t *>(sink.edge_index)+at*2;
        indices[0]=source;indices[1]=receiver;
        reinterpret_cast<int64_t *>(sink.edge_kind)[at]=kind;
        reinterpret_cast<uint8_t *>(sink.edge_mask)[at]=1;
        auto *feature=reinterpret_cast<float *>(sink.edge_features)+at*4;
        feature[0]=float(rs%8-ss%8)/7;feature[1]=float(rs/8-ss/8)/7;
        feature[2]=float(old)/12;feature[3]=float(next)/12;
    }
};
__global__ void tactical_export_kernel(const ChenRulePosition *positions,const Node *nodes,
    const uint64_t *history,uint64_t history_count,uint64_t start,uint64_t count,uint64_t epoch,
    ChenTacticalSink sink,uint32_t *error) {
    const uint64_t row=blockIdx.x;
    if(row>=count) return;
    const int lane=int(threadIdx.x),E=int(sink.edge_slots);
    const uint64_t index=start+row;
    const auto p=positions[index];
    __shared__ int codes[64],objects[64],locations[32],kings[2],np,nk;
    __shared__ uint64_t masks[64],pins[2],checkers,occ;
    __shared__ int64_t counters[4];
    __shared__ float phase;
    __shared__ ChenRuleMove moves[256];
    auto *pieces=reinterpret_cast<float *>(sink.pieces)+row*TP*16;
    auto *squares=reinterpret_cast<float *>(sink.squares)+row*TS*12;
    auto *actions=reinterpret_cast<float *>(sink.actions)+row*TK*12;
    for(int i=lane;i<TP*16;i+=64) pieces[i]=0;
    for(int i=lane;i<TK*12;i+=64) actions[i]=0;
    for(int i=lane;i<TP;i+=64) {
        reinterpret_cast<uint8_t *>(sink.piece_mask)[row*TP+i]=0;
        reinterpret_cast<int64_t *>(sink.piece_squares)[row*TP+i]=-1;
    }
    for(int i=lane;i<TK;i+=64) {
        reinterpret_cast<uint8_t *>(sink.action_mask)[row*TK+i]=0;
        reinterpret_cast<int64_t *>(sink.action_ids)[row*TK+i]=-1;
        reinterpret_cast<float *>(sink.action_prior_cp)[row*TK+i]=0;
    }
    for(int i=lane;i<E;i+=64) {
        const uint64_t at=row*E+i;
        reinterpret_cast<int64_t *>(sink.edge_index)[at*2]=0;
        reinterpret_cast<int64_t *>(sink.edge_index)[at*2+1]=0;
        reinterpret_cast<int64_t *>(sink.edge_kind)[at]=0;
        reinterpret_cast<uint8_t *>(sink.edge_mask)[at]=0;
        for(int c=0;c<4;++c) reinterpret_cast<float *>(sink.edge_features)[at*4+c]=0;
    }
    if(!lane) {
        occ=occupied(p);np=0;
        pins[0]=topology_orient(symbolic_pinned(p,int(p.side)),p.side);
        pins[1]=topology_orient(symbolic_pinned(p,1-int(p.side)),p.side);
        const int king=__ffsll(static_cast<unsigned long long>(p.boards[p.side*6+5]))-1;
        checkers=topology_orient(attackers(p,king,1-int(p.side)),p.side);
        tactical_history(positions,nodes,history,history_count,index,counters);
        constexpr int weights[6]={0,1,1,2,4,0};int total=0;
        for(int c=0;c<12;++c) total+=weights[c%6]*__popcll(p.boards[c]);
        phase=float(total>24?24:total)/24;
        nk=int(legal_moves(p,moves,error));
        for(int i=1;i<nk;++i) {
            const auto move=moves[i];int j=i;
            while(j && replay_action(moves[j-1],p.side)>replay_action(move,p.side)) {
                moves[j]=moves[j-1];--j;
            }
            moves[j]=move;
        }
        for(int h=0;h<4;++h) reinterpret_cast<int64_t *>(sink.history)[row*4+h]=counters[h];
        reinterpret_cast<uint64_t *>(sink.position_hash)[row]=p.hash;
        reinterpret_cast<uint64_t *>(sink.node_id)[row]=index;
        reinterpret_cast<uint64_t *>(sink.root_epoch)[row]=epoch;
        reinterpret_cast<uint8_t *>(sink.edge_overflow)[row]=0;
        reinterpret_cast<float *>(sink.value_prior_cp)[row]=float(evaluate(p,0));
        const int term=!nk?(checkers?1:2):p.halfmove>=100?3:counters[1]>=3?4:insufficient(p)?5:0;
        reinterpret_cast<int32_t *>(sink.terminal)[row]=term;
    }
    __syncthreads();
    const int raw=lane^(p.side?56:0),piece=piece_at(p,raw);
    codes[lane]=tactical_code(p,lane);
    masks[lane]=piece<0?0:topology_orient(symbolic_outgoing(p,raw,piece/6,occ),p.side);
    __syncthreads();
    if(!lane) for(int s=0;s<64;++s) {
        objects[s]=-1;
        if(codes[s]) {
            objects[s]=np;locations[np++]=s;
            if(codes[s]==6) kings[0]=s;
            if(codes[s]==12) kings[1]=s;
        }
    }
    __syncthreads();
    constexpr float scales[4]={100,3,256,256};
    for(int color=0;color<2;++color) {
        int incoming=0;
        for(int s=0;s<64;++s) if(codes[s] && (codes[s]>6)==bool(color)) incoming+=int((masks[s]>>lane)&1);
        const int outgoing=codes[lane] && (codes[lane]>6)==bool(color)?__popcll(masks[lane]):0;
        squares[lane*12+color]=float(incoming>8?8:incoming)/8;
        squares[lane*12+2+color]=float(outgoing>8?8:outgoing)/8;
        squares[lane*12+4+color]=float((pins[color]>>lane)&1);
    }
    squares[lane*12+6]=float((checkers>>lane)&1);squares[lane*12+7]=phase;
    for(int h=0;h<4;++h) squares[lane*12+8+h]=fminf(float(counters[h])/scales[h],1);
    if(codes[lane]) {
        const int slot=objects[lane];
        pieces[slot*16+codes[lane]-1]=1;pieces[slot*16+12]=float(lane%8)/7;
        pieces[slot*16+13]=float(lane/8)/7;
        pieces[slot*16+14]=float(((pins[0]|pins[1])>>lane)&1);
        pieces[slot*16+15]=float((checkers>>lane)&1);
        reinterpret_cast<uint8_t *>(sink.piece_mask)[row*TP+slot]=1;
        reinterpret_cast<int64_t *>(sink.piece_squares)[row*TP+slot]=lane;
    }
    __syncthreads();
    if(lane) return;
    TacticalWriter writer{sink,row,0,error};
    for(int s=0;s<64;++s) if(codes[s]) {
        const int object=objects[s],color=int(codes[s]>6);
        writer.add(object,TP+s,0,s,s,codes[s],codes[s]);
        writer.add(TP+s,object,1,s,s,codes[s],codes[s]);
        uint64_t mask=masks[s];
        while(mask) {
            const int target=pop(mask);
            writer.add(object,TP+target,2,s,target,codes[target],codes[target]);
            if(codes[target]) {
                const int relation=!color?(codes[target]<=6?4:3):(codes[target]<=6?5:6);
                writer.add(object,objects[target],relation,s,target,codes[target],codes[target]);
            }
        }
        if(((pins[color]>>s)&1)) {
            const int raw_pinner=topology_pinner(p,s^(p.side?56:0),color^int(p.side));
            const int pinner=raw_pinner^(p.side?56:0),king=kings[color];
            writer.add(object,objects[king],7,s,king,codes[king],codes[king]);
            writer.add(objects[pinner],object,8,pinner,s,codes[s],codes[s]);
        }
        if((checkers>>s)&1) writer.add(object,objects[kings[0]],9,s,kings[0],6,6);
    }
    constexpr int values[6]={100,320,330,500,900,0};
    for(int a=0;a<nk;++a) {
        const auto move=moves[a];
        const int source=move.from^(p.side?56:0),target=move.to^(p.side?56:0);
        const int kind=(codes[source]-1)%6;
        const bool ep=kind==0 && p.ep_square==move.to && !codes[target];
        const bool castle=kind==5 && abs(int(move.to)-int(move.from))==2;
        const int victim=ep?(int(move.to)+(p.side?8:-8))^(p.side?56:0):codes[target]?target:-1;
        const auto child=apply<false>(p,move);
        const int cp=-evaluate(child,0);
        auto *feature=actions+a*12;
        feature[0]=float(source%8)/7;feature[1]=float(source/8)/7;
        feature[2]=float(target%8)/7;feature[3]=float(target/8)/7;
        feature[4]=float(victim>=0);feature[5]=float(ep);feature[6]=float(castle);
        feature[7]=float(in_check(child,child.side));feature[8]=float(move.promotion)/4;
        feature[9]=victim<0?0:float(values[(codes[victim]-1)%6])/900;
        feature[10]=float(values[kind])/900;feature[11]=float(cp)/32000;
        reinterpret_cast<float *>(sink.action_prior_cp)[row*TK+a]=float(cp);
        reinterpret_cast<uint8_t *>(sink.action_mask)[row*TK+a]=1;
        reinterpret_cast<int64_t *>(sink.action_ids)[row*TK+a]=replay_action(move,p.side);
        const int node=TP+TS+a;
        writer.add(objects[source],node,10,source,target,codes[target],codes[target]);
        writer.add(TP+target,node,11,target,target,codes[target],codes[target]);
        if(victim>=0) writer.add(objects[victim],node,12,victim,target,codes[target],codes[target]);
        int edits[4]={source,target,-1,-1},n=2;
        if(ep) edits[n++]=victim;
        if(castle) {
            edits[n++]=(move.to>move.from?move.from+3:move.from-4)^(p.side?56:0);
            edits[n++]=(move.to>move.from?move.from+1:move.from-1)^(p.side?56:0);
        }
        for(int i=1;i<n;++i) { const int s=edits[i];int j=i;while(j && edits[j-1]>s) { edits[j]=edits[j-1];--j; }edits[j]=s; }
        for(int i=0;i<n;++i) {
            const int s=edits[i],raw_s=s^(p.side?56:0),next=piece_at(child,raw_s);
            const int code=next<0?0:next%6+1+((next/6)^int(p.side))*6;
            writer.add(node,TP+s,13,target,s,codes[s],code);
        }
    }
    auto *counts=reinterpret_cast<int32_t *>(sink.slot_counts)+row*3;
    counts[0]=np;counts[1]=nk;counts[2]=int(writer.edges);
}
}
extern "C" uint32_t chen_tactical_abi_version() { return 1; }
extern "C" int32_t chen_rules_export_tactical(ChenRulesContext *ctx,uint64_t start,uint64_t count,
    uint64_t epoch,const ChenTacticalSink *sink,ChenSymbolicExport *out,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !sink || !out || !range_valid(ctx,start,count) || ctx->mode==0)
        return fail(error,cap,"invalid tactical range or non-search arena");
    if(epoch!=ctx->root_epoch) return fail(error,cap,"stale tactical root epoch");
    if(sink->abi_version!=1 || sink->struct_size!=sizeof(*sink) || sink->reserved ||
        sink->device!=ctx->device || sink->capacity>uint64_t(INT32_MAX) ||
        sink->legal_slots!=TK || sink->edge_slots<4992 || sink->edge_slots>65536)
        return fail(error,cap,"invalid tactical schema/device/capacity");
    hipError_t code=hipSetDevice(ctx->device);
    if(code!=hipSuccess) return hip_fail(error,cap,code);
    const uint64_t addresses[]={sink->pieces,sink->squares,sink->actions,sink->piece_mask,sink->piece_squares,
        sink->action_mask,sink->action_ids,sink->edge_index,sink->edge_kind,sink->edge_mask,sink->edge_features,
        sink->edge_overflow,sink->value_prior_cp,sink->action_prior_cp,sink->position_hash,sink->node_id,
        sink->root_epoch,sink->history,sink->terminal,sink->slot_counts};
    const uint64_t E=sink->edge_slots;
    const uint64_t strides[]={TP*16*4,TS*12*4,TK*12*4,TP,TP*8,TK,TK*8,E*2*8,E*8,E,E*4*4,1,
        4,TK*4,8,8,8,4*8,4,3*4};
    const uint64_t aligns[]={4,4,4,1,8,1,8,8,8,1,4,1,4,4,8,8,8,8,4,4};
    int32_t status=export_intervals_valid(ctx,addresses,strides,aligns,20,sink->capacity,error,cap);
    if(status) return status;
    if(count>sink->capacity) return fail(error,cap,"tactical sink capacity exceeded",CHEN_RULES_CAPACITY);
    if(!count) { *out=ChenSymbolicExport{0,epoch,0};return 0; }
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(symbolic_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,start,count,ctx->used,&ctx->summary->error);
    hipLaunchKernelGGL(tactical_piece_preflight,grid(count),dim3(64),0,ctx->stream,
        ctx->positions,start,count,&ctx->summary->error);
    double preflight_ms=0;status=complete(ctx,&preflight_ms,error,cap);if(status) return status;
    status=prepare(ctx,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->stream);return status; }
    hipLaunchKernelGGL(tactical_export_kernel,dim3(uint32_t(count)),dim3(64),0,ctx->stream,
        ctx->positions,ctx->nodes,ctx->history,ctx->history_count,start,count,epoch,*sink,&ctx->summary->error);
    double export_ms=0;status=complete(ctx,&export_ms,error,cap);if(status) return status;
    *out=ChenSymbolicExport{count,epoch,preflight_ms+export_ms};return 0;
}
