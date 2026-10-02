// Included only by chen_rules.hip: shares exact device rules and evaluator code.
#include "chen_selective.h"
namespace {
struct SelectMeta { uint64_t open; uint32_t minimum,maximum; };
struct SelectHead { uint32_t selected,examined,added,stop; };
// Experimental classical-only sparse-ply backup. Each admitted parent has one
// ancestor at each ply, so a bucket needs at most MAX_BATCH distinct indices.
// Storage is part of the existing owned/alias-registered work allocation.
struct SelectSparseBackup {
    uint64_t nodes[CHEN_SELECTIVE_MAX_PLY][CHEN_SELECTIVE_MAX_BATCH];
    uint32_t counts[CHEN_SELECTIVE_MAX_PLY];
};
constexpr uint32_t SELECT_SPARSE_MARK=1u<<31;
static_assert(CHEN_SELECTIVE_MAX_CANDIDATES<SELECT_SPARSE_MARK,"reservation/touched bits disjoint");
static_assert(sizeof(SelectSparseBackup)==CHEN_SELECTIVE_MAX_PLY*(CHEN_SELECTIVE_MAX_BATCH*8+4),
    "bounded sparse-ply scratch size");
struct SelectWork {
    SelectHead head;
    uint64_t selected[CHEN_SELECTIVE_MAX_BATCH],offsets[CHEN_SELECTIVE_MAX_BATCH];
    SelectSparseBackup sparse;
};
struct SelectCompletion { ChenSelectiveStats stats;uint32_t error,reserved; };
static_assert(sizeof(SelectCompletion)==104,"private selective completion packet");
__global__ void selective_root_meta(const Node *nodes,SelectMeta *meta,uint32_t classified) {
    meta[0]=SelectMeta{classified && !(nodes[0].flags&TERMINAL)?1ull:0ull,0,0};
}
__device__ void selective_choose_device(const Node *nodes,const uint32_t *counts,const SelectMeta *meta,
    uint32_t *reserved,SelectWork *work,uint64_t used,uint64_t capacity,uint64_t remaining,
    uint32_t batch,uint32_t candidate_limit,uint32_t max_ply,uint32_t *error) {
    SelectHead head{0,0,0,CHEN_SELECTIVE_PROGRESS};
    bool budget_rejected=false,capacity_rejected=false;
    if(!meta[0].open) { head.stop=CHEN_SELECTIVE_CLOSED;work->head=head;return; }
    if(!remaining) { head.stop=CHEN_SELECTIVE_NODE_BUDGET;work->head=head;return; }
    if(used==capacity) { head.stop=CHEN_SELECTIVE_ARENA_CAPACITY;work->head=head;return; }
    while(head.examined<candidate_limit && head.selected<batch && meta[0].open>reserved[0]) {
        uint64_t index=0;
        while(nodes[index].flags&EXPANDED) {
            const Node node=nodes[index];
            int best=-MATE-1;uint64_t next=NONE;
            for(uint32_t j=0;j<node.child_count;++j) {
                const uint64_t child=node.first_child+j;
                if(child>=used) { device_error(error,4);return; }
                if(meta[child].open>reserved[child] && -nodes[child].score>best) {
                    best=-nodes[child].score;next=child;
                }
            }
            if(next==NONE) { device_error(error,4);return; }
            index=next;
        }
        const Node leaf=nodes[index];
        if(!(leaf.flags&CLASSIFIED) || (leaf.flags&TERMINAL) || leaf.ply>=max_ply ||
            (index && !(leaf.flags&SCORED)) || !counts[index] || reserved[index]) {
            device_error(error,4);return;
        }
        ++head.examined;
        for(uint64_t parent=index;parent!=NONE;parent=nodes[parent].parent) {
            if(parent>=used || reserved[parent]>=candidate_limit) { device_error(error,4);return; }
            ++reserved[parent];
        }
        const uint64_t cost=counts[index];
        const bool node_fits=cost<=remaining-head.added;
        const bool arena_fits=cost<=capacity-used-head.added;
        budget_rejected |= !node_fits;capacity_rejected |= !arena_fits;
        if(node_fits && arena_fits) {
            work->selected[head.selected]=index;work->offsets[head.selected]=head.added;
            ++head.selected;head.added+=uint32_t(cost);
        }
    }
    if(!head.selected) {
        if(head.examined==candidate_limit && meta[0].open>reserved[0]) head.stop=CHEN_SELECTIVE_CANDIDATE_SCAN;
        else if(budget_rejected) head.stop=CHEN_SELECTIVE_NODE_BUDGET;
        else if(capacity_rejected) head.stop=CHEN_SELECTIVE_ARENA_CAPACITY;
        else { device_error(error,4);return; }
    }
    work->head=head;
}
__global__ void selective_choose(const Node *nodes,const uint32_t *counts,const SelectMeta *meta,
    uint32_t *reserved,SelectWork *work,uint64_t used,uint64_t capacity,uint64_t remaining,
    uint32_t batch,uint32_t candidate_limit,uint32_t max_ply,uint32_t *error) {
    selective_choose_device(nodes,counts,meta,reserved,work,used,capacity,remaining,
        batch,candidate_limit,max_ply,error);
}
__global__ void selective_expand(ChenRulePosition *positions,Node *nodes,const ChenRuleMove *moves,
    const uint32_t *counts,const SelectWork *work,uint64_t start) {
    const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=work->head.selected) return;
    const uint64_t index=work->selected[row],first=start+work->offsets[row];
    const Node parent=nodes[index];
    for(uint32_t j=0;j<counts[index];++j) {
        const ChenRuleMove move=moves[index*CHEN_RULES_MAX_MOVES+j];
        positions[first+j]=apply<true>(positions[index],move);
        nodes[first+j]=Node{index,0,NONE,0,parent.ply+1,0,0,move,0};
    }
    nodes[index].first_child=first;nodes[index].child_count=counts[index];
    nodes[index].flags|=EXPANDED;
}
__global__ void selective_new_meta(const Node *nodes,SelectMeta *meta,uint64_t start,uint64_t count,
    uint32_t max_ply,uint32_t *error) {
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=count) return;
    const uint64_t index=start+row;const Node node=nodes[index];
    if(!(node.flags&CLASSIFIED) || !(node.flags&SCORED) || node.ply>max_ply) {
        device_error(error,4);return;
    }
    meta[index]=SelectMeta{!(node.flags&TERMINAL) && node.ply<max_ply?1ull:0ull,node.ply,node.ply};
}
__device__ void selective_backup_device(Node *nodes,SelectMeta *meta,const uint32_t *reserved,
    uint64_t old_used,uint64_t used,uint32_t *error) {
    // Children always have larger indices. One ordered device pass is the
    // reference implementation; no parent/child parallel-write race is possible.
    for(uint64_t cursor=old_used;cursor;) {
        const uint64_t index=--cursor;
        if(!reserved[index] || !(nodes[index].flags&EXPANDED)) continue;
        Node &node=nodes[index];int best=-MATE-1;uint64_t winner=NONE;
        SelectMeta aggregate{0,UINT32_MAX,0};
        for(uint32_t j=0;j<node.child_count;++j) {
            const uint64_t child=node.first_child+j;
            if(child>=used || !(nodes[child].flags&SCORED)) { device_error(error,4);return; }
            const int score=-nodes[child].score;
            if(score>best) { best=score;winner=child; }
            aggregate.open+=meta[child].open;
            if(meta[child].minimum<aggregate.minimum) aggregate.minimum=meta[child].minimum;
            if(meta[child].maximum>aggregate.maximum) aggregate.maximum=meta[child].maximum;
        }
        if(winner==NONE) { device_error(error,4);return; }
        node.score=best;node.best=winner;node.flags|=SCORED;meta[index]=aggregate;
    }
}
__global__ void selective_backup(Node *nodes,SelectMeta *meta,const uint32_t *reserved,
    uint64_t old_used,uint64_t used,uint32_t *error) {
    selective_backup_device(nodes,meta,reserved,old_used,used,error);
}
__device__ void selective_backup_sparse_device(Node *nodes,SelectMeta *meta,uint32_t *reserved,
    SelectWork *work,uint64_t old_used,uint64_t used,uint32_t max_ply,uint32_t *error) {
    // Called only after classical selection and complete child initialization.
    // Rejected candidates cannot change values/metadata, so visit ancestors of
    // admitted expansions only. The high bit is never read by selection: its
    // reservation scratch is cleared before the next advance, as before.
    if(!max_ply || max_ply>CHEN_SELECTIVE_MAX_PLY ||
        work->head.selected>CHEN_SELECTIVE_MAX_BATCH) { device_error(error,4);return; }
    for(uint32_t ply=0;ply<max_ply;++ply) work->sparse.counts[ply]=0;
    for(uint32_t row=0;row<work->head.selected;++row) {
        uint64_t index=work->selected[row];
        for(uint32_t depth=0;index!=NONE;++depth) {
            if(index>=old_used || depth>=max_ply) { device_error(error,4);return; }
            if(reserved[index]&SELECT_SPARSE_MARK) break;
            const Node node=nodes[index];
            if(!(node.flags&EXPANDED) || !reserved[index] || node.ply>=max_ply ||
                (index==0 && (node.parent!=NONE || node.ply)) ||
                (index && (node.parent>=index || nodes[node.parent].ply+1!=node.ply))) {
                device_error(error,4);return;
            }
            uint32_t &count=work->sparse.counts[node.ply];
            if(count>=work->head.selected || count>=CHEN_SELECTIVE_MAX_BATCH) {
                device_error(error,4);return;
            }
            work->sparse.nodes[node.ply][count++]=index;
            reserved[index]|=SELECT_SPARSE_MARK;
            index=node.parent;
        }
    }
    // Every dependency's child ply is greater than its parent's. Nodes at one
    // ply are independent; bucket insertion order cannot affect integer scores
    // or first-legal-child ties. One thread keeps this reference variant race
    // free without sorting the arena or launching MAX_PLY separate kernels.
    for(uint32_t cursor=max_ply;cursor;) {
        const uint32_t ply=--cursor;
        for(uint32_t row=0;row<work->sparse.counts[ply];++row) {
            const uint64_t index=work->sparse.nodes[ply][row];
            Node &node=nodes[index];int best=-MATE-1;uint64_t winner=NONE;
            SelectMeta aggregate{0,UINT32_MAX,0};
            for(uint32_t j=0;j<node.child_count;++j) {
                const uint64_t child=node.first_child+j;
                if(child>=used || !(nodes[child].flags&SCORED)) { device_error(error,4);return; }
                const int score=-nodes[child].score;
                if(score>best) { best=score;winner=child; }
                aggregate.open+=meta[child].open;
                if(meta[child].minimum<aggregate.minimum) aggregate.minimum=meta[child].minimum;
                if(meta[child].maximum>aggregate.maximum) aggregate.maximum=meta[child].maximum;
            }
            if(winner==NONE) { device_error(error,4);return; }
            node.score=best;node.best=winner;node.flags|=SCORED;meta[index]=aggregate;
        }
    }
}
__device__ void selective_snapshot_device(const Node *nodes,const SelectMeta *meta,
    const unsigned long long *classified,const unsigned long long *evaluated,
    ChenSelectiveStats *out,uint64_t used,uint64_t steps,uint64_t expanded,
    uint32_t selected,uint32_t added,uint32_t stop,uint32_t max_ply,uint32_t *error) {
    const Node root=nodes[0];uint32_t flags=0,pv=0;
    if(root.flags&SCORED) {
        if(!(root.flags&(TERMINAL|EXPANDED))) { device_error(error,4);return; }
        flags|=CHEN_SELECTIVE_ROOT_PUBLISHED;
        if(root.flags&TERMINAL) flags|=CHEN_SELECTIVE_ROOT_TERMINAL;
        if(!meta[0].open) flags|=CHEN_SELECTIVE_HORIZON_COMPLETE;
        if(root.score>MAX_EVAL || root.score<-MAX_EVAL) flags|=CHEN_SELECTIVE_MATE_PROOF;
        uint64_t index=0;
        while(nodes[index].best!=NONE) {
            index=nodes[index].best;
            if(index>=used || ++pv>max_ply) { device_error(error,4);return; }
        }
    }
    *out=ChenSelectiveStats{1,sizeof(ChenSelectiveStats),used,uint64_t(*classified),uint64_t(*evaluated),
        steps,expanded,meta[0].open,selected,added,0,meta[0].minimum,meta[0].maximum,pv,flags,stop,0};
}
__global__ void selective_snapshot(const Node *nodes,const SelectMeta *meta,
    const unsigned long long *classified,const unsigned long long *evaluated,
    ChenSelectiveStats *out,uint64_t used,uint64_t steps,uint64_t expanded,
    uint32_t selected,uint32_t added,uint32_t stop,uint32_t max_ply,uint32_t *error) {
    selective_snapshot_device(nodes,meta,classified,evaluated,out,used,steps,expanded,
        selected,added,stop,max_ply,error);
}

__global__ void selective_choose_classical(const Node *nodes,const uint32_t *counts,SelectMeta *meta,
    const unsigned long long *classified,uint32_t *reserved,SelectWork *work,
    uint64_t used,uint64_t capacity,uint64_t limit,uint32_t bootstrap,
    uint32_t batch,uint32_t candidate_limit,uint32_t max_ply,uint32_t *error) {
    work->head=SelectHead{0,0,0,CHEN_SELECTIVE_PROGRESS};
    if(*error) return;
    if(bootstrap) {
        if(!limit) { work->head.stop=CHEN_SELECTIVE_NODE_BUDGET;return; }
        if(!(nodes[0].flags&CLASSIFIED)) { device_error(error,4);return; }
        meta[0]=SelectMeta{!(nodes[0].flags&TERMINAL)?1ull:0ull,0,0};
    }
    if(*classified>limit) { device_error(error,4);return; }
    selective_choose_device(nodes,counts,meta,reserved,work,used,capacity,limit-*classified,
        batch,candidate_limit,max_ply,error);
}

__global__ void selective_children_classical(ChenRulePosition *positions,Node *nodes,
    ChenRuleMove *moves,uint32_t *counts,SelectMeta *meta,const SelectWork *work,
    uint64_t start,uint64_t capacity,uint64_t limit,uint32_t batch,uint32_t candidate_limit,
    uint32_t max_ply,uint32_t evaluator,const uint64_t *history,uint64_t history_count,
    unsigned long long *classified,unsigned long long *evaluated,uint32_t *error) {
    if(*error) return;
    const SelectHead head=work->head;
    if(head.selected>batch || head.examined>candidate_limit || head.added<head.selected ||
        head.added>capacity-start || (head.added && (start>limit || head.added>limit-start)) ||
        head.stop>CHEN_SELECTIVE_FALLBACK || (!head.selected && head.added)) {
        device_error(error,4);return;
    }
    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=head.added) return;
    // Prefix offsets are strictly increasing because every selected legal set
    // is nonempty. Find the last selected offset <= this child's block offset.
    uint32_t lo=0,hi=head.selected;
    while(lo+1<hi) {
        const uint32_t mid=lo+(hi-lo)/2;
        if(work->offsets[mid]<=row) lo=mid;else hi=mid;
    }
    const uint64_t parent=work->selected[lo],offset=work->offsets[lo];
    if(parent>=start || offset>row || row-offset>=counts[parent]) { device_error(error,4);return; }
    const uint32_t child=uint32_t(row-offset),parent_ply=nodes[parent].ply;
    if(parent_ply>=max_ply) { device_error(error,4);return; }
    const uint64_t index=start+row;
    const ChenRuleMove move=moves[parent*CHEN_RULES_MAX_MOVES+child];
    const ChenRulePosition p=apply<true>(positions[parent],move);
    positions[index]=p;
    nodes[index]=Node{parent,0,NONE,0,parent_ply+1,0,0,move,0};
    // Read ONLY immutable parent ply/position fields above. Child0 publishes
    // expansion fields concurrently with siblings; no whole-parent Node read.
    if(!child) {
        nodes[parent].first_child=start+offset;nodes[parent].child_count=counts[parent];
        nodes[parent].flags|=EXPANDED;
    }
    Node &node=nodes[index];
    // Exact classify_kernel terminal logic, duplicated locally to fuse apply,
    // legal generation, classification, integer evaluation and leaf metadata.
    counts[index]=legal_moves(p,moves+index*CHEN_RULES_MAX_MOVES,error);
    node.flags=(in_check(p,p.side)?CHECK:0)|(insufficient(p)?INSUFFICIENT:0)|CLASSIFIED;
    atomicAdd(classified,1ull);
    if(!counts[index]) {
        node.score=(node.flags&CHECK)?-MATE+int(node.ply):0;node.flags|=TERMINAL|SCORED;
    } else if(p.halfmove>=100 || (node.flags&INSUFFICIENT) ||
        repeated(positions,nodes,index,history,history_count)) {
        node.score=0;node.flags|=TERMINAL|SCORED;
    }
    if(!(node.flags&SCORED)) {
        node.score=evaluate(p,evaluator);node.flags|=SCORED;atomicAdd(evaluated,1ull);
    }
    meta[index]=SelectMeta{!(node.flags&TERMINAL) && node.ply<max_ply?1ull:0ull,node.ply,node.ply};
}

__global__ void selective_complete_classical(Node *nodes,SelectMeta *meta,uint32_t *reserved,
    SelectWork *work,const unsigned long long *classified,const unsigned long long *evaluated,
    SelectCompletion *out,uint64_t old_used,uint64_t steps,uint64_t expanded,
    uint32_t max_ply,uint32_t *error) {
    *out=SelectCompletion{};
    if(!*error) {
        const SelectHead head=work->head;
        const uint64_t used=old_used+head.added;
        if(*classified!=used && !(*classified==0 && used==1)) device_error(error,4);
        if(head.selected && !*error)
            selective_backup_sparse_device(nodes,meta,reserved,work,old_used,used,max_ply,error);
        if(!*error) {
            const uint32_t stop=!(meta[0].open) && (nodes[0].flags&SCORED)?CHEN_SELECTIVE_CLOSED:head.stop;
            selective_snapshot_device(nodes,meta,classified,evaluated,&out->stats,used,
                steps+(head.selected?1:0),expanded+head.selected,
                head.selected,head.added,stop,max_ply,error);
        }
    }
    out->error=*error;
}
}
struct ChenSelectiveContext {
    ChenRulesContext *arena=nullptr;SelectMeta *meta=nullptr;SelectWork *work=nullptr;
    ChenSelectiveStats *snapshot=nullptr;SelectCompletion *host_completion=nullptr;
    void *mapped_completion=nullptr;ChenSelectiveConfig config{};
    bool valid=false,root_classified=false;int evaluator=-1;
    ChenExternalEvaluator external{};
    uint64_t classified=0,steps=0,expanded=0;
    uint32_t selected=0,added=0,stop=CHEN_SELECTIVE_FALLBACK;
};
namespace {
int32_t selective_error(ChenSelectiveContext *ctx,char *error,uint64_t cap,const char *message) {
    if(ctx) ctx->valid=false;
    return fail(error,cap,message);
}
int32_t selective_status(ChenSelectiveContext *ctx,int32_t status) {
    if(status) ctx->valid=false;
    return status;
}
int32_t selective_classical(ChenSelectiveContext *ctx,uint64_t limit,uint32_t evaluator,
    ChenSelectiveStats *stats,char *error,uint64_t cap) {
    ChenRulesContext *arena=ctx->arena;
    const uint64_t old_used=arena->used;
    int32_t status=lock_mode(arena,1,error,cap);
    if(status) return selective_status(ctx,status);
    if(arena->evaluator>=0 && arena->evaluator!=int(evaluator))
        return selective_error(ctx,error,cap,"resident evaluator cannot change before reset");
    arena->evaluator=int(evaluator);
    status=prepare(arena,error,cap);
    if(status) { (void)hipStreamSynchronize(arena->stream);return selective_status(ctx,status); }
    hipError_t code=hipMemsetAsync(arena->flags,0,old_used*sizeof(*arena->flags),arena->stream);
    if(code!=hipSuccess) return selective_status(ctx,transfer_failure(arena,code,error,cap));
    if(!ctx->root_classified && limit) {
        hipLaunchKernelGGL(classify_kernel,dim3(1),dim3(64),0,arena->stream,
            arena->positions,arena->nodes,arena->moves,arena->counts,0ull,1ull,1u,
            arena->history,arena->history_count,arena->classified,&arena->summary->error);
        code=hipGetLastError();
        if(code!=hipSuccess) return selective_status(ctx,transfer_failure(arena,code,error,cap));
    }
    hipLaunchKernelGGL(selective_choose_classical,dim3(1),dim3(1),0,arena->stream,
        arena->nodes,arena->counts,ctx->meta,arena->classified,arena->flags,ctx->work,
        old_used,arena->capacity,limit,uint32_t(!ctx->root_classified),
        ctx->config.batch_leaves,ctx->config.candidate_limit,ctx->config.max_ply,&arena->summary->error);
    code=hipGetLastError();
    if(code!=hipSuccess) return selective_status(ctx,transfer_failure(arena,code,error,cap));
    uint64_t bound=uint64_t(ctx->config.batch_leaves)*CHEN_RULES_MAX_MOVES;
    if(bound>arena->capacity-old_used) bound=arena->capacity-old_used;
    if(bound>limit-ctx->classified) bound=limit-ctx->classified;
    // Admission count stays on the device; bounded excess threads return early.
    hipLaunchKernelGGL(selective_children_classical,grid(bound?bound:1),dim3(64),0,arena->stream,
        arena->positions,arena->nodes,arena->moves,arena->counts,ctx->meta,ctx->work,
        old_used,arena->capacity,limit,ctx->config.batch_leaves,ctx->config.candidate_limit,
        ctx->config.max_ply,evaluator,arena->history,arena->history_count,
        arena->classified,arena->evaluated,&arena->summary->error);
    code=hipGetLastError();
    if(code!=hipSuccess) return selective_status(ctx,transfer_failure(arena,code,error,cap));
    hipLaunchKernelGGL(selective_complete_classical,dim3(1),dim3(1),0,arena->stream,
        arena->nodes,ctx->meta,arena->flags,ctx->work,arena->classified,arena->evaluated,
        reinterpret_cast<SelectCompletion *>(ctx->snapshot),old_used,ctx->steps,ctx->expanded,
        ctx->config.max_ply,&arena->summary->error);
    code=hipGetLastError();
    if(code!=hipSuccess) return selective_status(ctx,transfer_failure(arena,code,error,cap));
    code=hipEventRecord(arena->end,arena->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->host_completion,ctx->snapshot,
        sizeof(SelectCompletion),hipMemcpyDeviceToHost,arena->stream);
    const hipError_t sync=hipStreamSynchronize(arena->stream);
    if(code==hipSuccess) code=sync;
    float elapsed=0;
    if(code==hipSuccess) code=hipEventElapsedTime(&elapsed,arena->begin,arena->end);
    if(code!=hipSuccess) return selective_status(ctx,hip_fail(error,cap,code));
    // Completion is the only successful-path fence and scalar read in advance.
    const SelectCompletion &packet=*ctx->host_completion;
    if(packet.error) {
        const char *message=packet.error==1?"invalid rules position or mismatched hash":
            packet.error==2?"rules arena/move capacity exceeded":
            "invalid resident tree stage or incomplete child scores";
        ctx->valid=false;
        return fail(error,cap,message,packet.error==2?CHEN_RULES_CAPACITY:-1);
    }
    const ChenSelectiveStats &s=packet.stats;
    if(packet.reserved || s.abi_version!=1 || s.struct_size!=sizeof(s) || s.reserved ||
        s.last_selected>ctx->config.batch_leaves || s.last_added<s.last_selected ||
        s.arena_positions!=old_used+s.last_added || s.arena_positions>arena->capacity ||
        s.classified_positions>limit || s.classified_positions<s.last_added ||
        s.classified_positions<ctx->classified || s.evaluated_positions>s.classified_positions ||
        (s.classified_positions!=s.arena_positions && !(s.classified_positions==0 && s.arena_positions==1)) ||
        s.completed_steps!=ctx->steps+(s.last_selected?1:0) ||
        s.expanded_leaves!=ctx->expanded+s.last_selected || s.stop_reason>CHEN_SELECTIVE_FALLBACK)
        return selective_error(ctx,error,cap,"invalid device classical completion packet");
    arena->kernel_total_ms+=elapsed;arena->used=s.arena_positions;
    ctx->root_classified=s.classified_positions!=0;ctx->classified=s.classified_positions;
    ctx->steps=s.completed_steps;ctx->expanded=s.expanded_leaves;
    ctx->selected=s.last_selected;ctx->added=s.last_added;ctx->stop=s.stop_reason;
    *stats=s;stats->kernel_ms=arena->kernel_total_ms;
    return 0;
}
int32_t selective_stats(ChenSelectiveContext *ctx,ChenSelectiveStats *out,char *error,uint64_t cap) {
    int32_t status=prepare(ctx->arena,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->arena->stream);return selective_status(ctx,status); }
    hipLaunchKernelGGL(selective_snapshot,dim3(1),dim3(1),0,ctx->arena->stream,
        ctx->arena->nodes,ctx->meta,ctx->arena->classified,ctx->arena->evaluated,ctx->snapshot,
        ctx->arena->used,ctx->steps,ctx->expanded,ctx->selected,ctx->added,ctx->stop,
        ctx->config.max_ply,&ctx->arena->summary->error);
    status=complete(ctx->arena,nullptr,error,cap);if(status) return selective_status(ctx,status);
    const hipError_t code=hipMemcpy(out,ctx->snapshot,sizeof(*out),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return selective_status(ctx,hip_fail(error,cap,code));
    out->kernel_ms=ctx->arena->kernel_total_ms;
    return 0;
}
void selective_retire(hipError_t code) {
    if(code==hipSuccess) return;
    std::fprintf(stderr,"chen selective: unproven GPU retirement: %s\n",hipGetErrorString(code));
    std::fflush(stderr);std::_Exit(70);
}
}
extern "C" uint32_t chen_selective_abi_version() { return 1; }
extern "C" void chen_selective_destroy(ChenSelectiveContext *ctx) {
    if(!ctx) return;
    if(ctx->arena) {
        selective_retire(hipSetDevice(ctx->arena->device));
        selective_retire(hipStreamSynchronize(ctx->arena->stream));
        if(ctx->meta) selective_retire(hipFree(ctx->meta));
        if(ctx->work) selective_retire(hipFree(ctx->work));
        if(ctx->snapshot) selective_retire(hipFree(ctx->snapshot));
        if(ctx->host_completion) selective_retire(hipHostFree(ctx->host_completion));
        for(uint32_t i=0;i<4;++i) { ctx->arena->extra_private[i]=nullptr;ctx->arena->extra_private_bytes[i]=0; }
        chen_rules_destroy(ctx->arena);
    }
    delete ctx;
}
extern "C" ChenSelectiveContext *chen_selective_create(int32_t device,uint64_t capacity,
    char *error,uint64_t cap) {
    clear_error(error,cap);
    auto *ctx=new(std::nothrow) ChenSelectiveContext;
    if(!ctx) { fail(error,cap,"selective context allocation failed");return nullptr; }
    ctx->arena=chen_rules_create(device,capacity,error,cap);
    if(!ctx->arena) { delete ctx;return nullptr; }
    hipError_t code=hipSetDevice(device);
    if(code==hipSuccess) code=hipMalloc(&ctx->meta,capacity*sizeof(*ctx->meta));
    if(code==hipSuccess) code=hipMalloc(&ctx->work,sizeof(*ctx->work));
    if(code==hipSuccess) code=hipMalloc(&ctx->snapshot,sizeof(SelectCompletion));
    if(code==hipSuccess) code=hipHostMalloc(&ctx->host_completion,sizeof(SelectCompletion),hipHostMallocMapped);
    if(code==hipSuccess) code=hipHostGetDevicePointer(&ctx->mapped_completion,ctx->host_completion,0);
    if(code!=hipSuccess) { hip_fail(error,cap,code);chen_selective_destroy(ctx);return nullptr; }
    ctx->arena->extra_private[0]=ctx->meta;ctx->arena->extra_private_bytes[0]=capacity*sizeof(*ctx->meta);
    ctx->arena->extra_private[1]=ctx->work;ctx->arena->extra_private_bytes[1]=sizeof(*ctx->work);
    ctx->arena->extra_private[2]=ctx->snapshot;ctx->arena->extra_private_bytes[2]=sizeof(SelectCompletion);
    ctx->arena->extra_private[3]=ctx->mapped_completion;ctx->arena->extra_private_bytes[3]=sizeof(SelectCompletion);
    return ctx;
}
extern "C" int32_t chen_selective_reset(ChenSelectiveContext *ctx,const ChenRulePosition *root,
    const uint64_t *history,uint64_t history_count,const ChenSelectiveConfig *config,
    ChenSelectiveStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx) return fail(error,cap,"null selective context");
    ctx->valid=false;
    if(!root || !config || !stats || config->abi_version!=1 || config->struct_size!=sizeof(*config) ||
        !config->max_ply || config->max_ply>CHEN_SELECTIVE_MAX_PLY ||
        config->policy!=CHEN_SELECTIVE_POLICY_BEST_OPEN || !config->batch_leaves ||
        config->batch_leaves>CHEN_SELECTIVE_MAX_BATCH || config->candidate_limit<config->batch_leaves ||
        config->candidate_limit>CHEN_SELECTIVE_MAX_CANDIDATES || config->reserved)
        return fail(error,cap,"invalid selective reset config/buffers");
    int32_t status=chen_rules_reset(ctx->arena,root,history,history_count,error,cap);
    if(status) return status;
    ctx->config=*config;ctx->root_classified=false;ctx->evaluator=-1;ctx->external={};
    ctx->classified=ctx->steps=ctx->expanded=0;ctx->selected=ctx->added=0;ctx->stop=CHEN_SELECTIVE_FALLBACK;
    status=prepare(ctx->arena,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->arena->stream);return status; }
    hipLaunchKernelGGL(selective_root_meta,dim3(1),dim3(1),0,ctx->arena->stream,ctx->arena->nodes,ctx->meta,0);
    status=complete(ctx->arena,nullptr,error,cap);if(status) return status;
    ctx->valid=true;return selective_stats(ctx,stats,error,cap);
}
extern "C" int32_t chen_selective_advance(ChenSelectiveContext *ctx,uint64_t limit,
    uint32_t evaluator,const ChenExternalEvaluator *external,ChenSelectiveStats *stats,
    char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !ctx->valid) return fail(error,cap,"selective context must reset before advance");
    if(!stats || limit<ctx->classified || evaluator>2 || (evaluator==2)!=bool(external))
        return selective_error(ctx,error,cap,"invalid selective limit/evaluator/buffers");
    if(ctx->evaluator>=0 && ctx->evaluator!=int(evaluator))
        return selective_error(ctx,error,cap,"selective evaluator changed before reset");
    if(external) {
        if(external->abi_version!=1 || external->struct_size!=sizeof(*external) ||
            !external->evaluate || !external->max_batch || external->max_batch>uint64_t(INT32_MAX) || external->reserved)
            return selective_error(ctx,error,cap,"invalid selective external descriptor");
        for(uint32_t i=0;i<5;++i) if(external->runtime_identity[i]!=chen_replay_runtime_identity(i))
            return selective_error(ctx,error,cap,"selective external runtime identity mismatch");
        if(ctx->evaluator==2 && (external->evaluate!=ctx->external.evaluate || external->owner!=ctx->external.owner ||
            external->max_batch!=ctx->external.max_batch))
            return selective_error(ctx,error,cap,"selective external identity changed before reset");
        ctx->external=*external;
    }
    ctx->evaluator=int(evaluator);ctx->selected=ctx->added=0;
    if(evaluator<2) return selective_classical(ctx,limit,evaluator,stats,error,cap);
    if(!ctx->root_classified) {
        if(!limit) { ctx->stop=CHEN_SELECTIVE_NODE_BUDGET;return selective_stats(ctx,stats,error,cap); }
        ChenRulesStats baseline{};
        int32_t status=chen_rules_classify(ctx->arena,0,1,1,&baseline,error,cap);
        if(status) return selective_status(ctx,status);
        ctx->root_classified=true;ctx->classified=1;
        status=prepare(ctx->arena,error,cap);
        if(status) { (void)hipStreamSynchronize(ctx->arena->stream);return selective_status(ctx,status); }
        hipLaunchKernelGGL(selective_root_meta,dim3(1),dim3(1),0,ctx->arena->stream,ctx->arena->nodes,ctx->meta,1);
        status=complete(ctx->arena,nullptr,error,cap);if(status) return selective_status(ctx,status);
    }
    const uint64_t old_used=ctx->arena->used;
    int32_t status=prepare(ctx->arena,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->arena->stream);return selective_status(ctx,status); }
    hipError_t code=hipMemsetAsync(ctx->arena->flags,0,old_used*sizeof(*ctx->arena->flags),ctx->arena->stream);
    if(code!=hipSuccess) return selective_status(ctx,transfer_failure(ctx->arena,code,error,cap));
    hipLaunchKernelGGL(selective_choose,dim3(1),dim3(1),0,ctx->arena->stream,
        ctx->arena->nodes,ctx->arena->counts,ctx->meta,ctx->arena->flags,ctx->work,old_used,
        ctx->arena->capacity,limit-ctx->classified,ctx->config.batch_leaves,ctx->config.candidate_limit,
        ctx->config.max_ply,&ctx->arena->summary->error);
    status=complete(ctx->arena,nullptr,error,cap);if(status) return selective_status(ctx,status);
    SelectHead head{};code=hipMemcpy(&head,ctx->work,sizeof(head),hipMemcpyDeviceToHost);
    if(code!=hipSuccess) return selective_status(ctx,hip_fail(error,cap,code));
    if(head.selected>ctx->config.batch_leaves || head.examined>ctx->config.candidate_limit ||
        head.added>ctx->arena->capacity-old_used || head.added>limit-ctx->classified ||
        head.stop>CHEN_SELECTIVE_FALLBACK || (!head.selected && head.added) || head.added<head.selected)
        return selective_error(ctx,error,cap,"invalid device selection summary");
    ctx->stop=head.stop;
    if(!head.selected) return selective_stats(ctx,stats,error,cap);
    status=prepare(ctx->arena,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->arena->stream);return selective_status(ctx,status); }
    hipLaunchKernelGGL(selective_expand,grid(head.selected),dim3(64),0,ctx->arena->stream,
        ctx->arena->positions,ctx->arena->nodes,ctx->arena->moves,ctx->arena->counts,ctx->work,old_used);
    status=complete(ctx->arena,nullptr,error,cap);if(status) return selective_status(ctx,status);
    ctx->arena->used+=head.added;
    ChenRulesStats evaluated{};
    status=evaluator==2?chen_rules_evaluate_external(ctx->arena,old_used,head.added,external,&evaluated,error,cap):
        chen_rules_evaluate(ctx->arena,old_used,head.added,evaluator,&evaluated,error,cap);
    if(status) return selective_status(ctx,status);
    if(evaluated.classified_positions!=ctx->arena->used)
        return selective_error(ctx,error,cap,"selective child classification incomplete");
    status=prepare(ctx->arena,error,cap);
    if(status) { (void)hipStreamSynchronize(ctx->arena->stream);return selective_status(ctx,status); }
    hipLaunchKernelGGL(selective_new_meta,grid(head.added),dim3(64),0,ctx->arena->stream,
        ctx->arena->nodes,ctx->meta,old_used,uint64_t(head.added),ctx->config.max_ply,&ctx->arena->summary->error);
    code=hipGetLastError();
    if(code!=hipSuccess) return selective_status(ctx,transfer_failure(ctx->arena,code,error,cap));
    hipLaunchKernelGGL(selective_backup,dim3(1),dim3(1),0,ctx->arena->stream,
        ctx->arena->nodes,ctx->meta,ctx->arena->flags,old_used,ctx->arena->used,&ctx->arena->summary->error);
    status=complete(ctx->arena,nullptr,error,cap);if(status) return selective_status(ctx,status);
    ctx->classified=evaluated.classified_positions;++ctx->steps;ctx->expanded+=head.selected;
    ctx->selected=head.selected;ctx->added=head.added;ctx->stop=CHEN_SELECTIVE_PROGRESS;
    status=selective_stats(ctx,stats,error,cap);
    if(!status && (stats->root_flags&CHEN_SELECTIVE_HORIZON_COMPLETE)) {
        ctx->stop=CHEN_SELECTIVE_CLOSED;stats->stop_reason=ctx->stop;
    }
    return status;
}
extern "C" int32_t chen_selective_finish(ChenSelectiveContext *ctx,ChenRulesResult *out,
    ChenRuleMove *pv,uint64_t pv_capacity,ChenSelectiveStats *stats,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!ctx || !ctx->valid || !out || !stats || (pv_capacity&&!pv))
        return fail(error,cap,"invalid selective finish buffers/context");
    int32_t status=selective_stats(ctx,stats,error,cap);if(status) return status;
    if(!(stats->root_flags&CHEN_SELECTIVE_ROOT_PUBLISHED))
        return fail(error,cap,"selective root has no completed legal support");
    ChenRulesStats baseline{};
    status=chen_rules_finish(ctx->arena,out,pv,pv_capacity,&baseline,error,cap);
    if(status && status!=CHEN_RULES_CAPACITY) ctx->valid=false;
    if(!status) stats->kernel_ms=ctx->arena->kernel_total_ms;
    return status;
}
