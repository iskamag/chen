// New ABI3 experiment only. Inclusion binds current production rules closure,
// including symbolic/tactical exports, without modifying any shared GPU source.
#include "bounded_search_timed.hip"
#include "root_pool.h"
namespace {
struct PoolState {
    ChenBoundedJob config;
    ChenRuleMove actions[CHEN_RULES_MAX_MOVES];
    ChenRootPoolChild children[CHEN_RULES_MAX_MOVES];
    ChenRootPoolSnapshot snapshot;
    uint64_t iteration_start_nodes,epoch_began,epoch_ticks;
    int32_t incumbent;
    uint32_t preferred,prime;
};
__device__ void pool_fail(PoolState &p,uint32_t reason) {
    atomicCAS(&p.snapshot.abort_reason,0u,reason);
}
__device__ bool pool_node(PoolState &p) {
    unsigned long long *counter=reinterpret_cast<unsigned long long *>(&p.snapshot.nodes);
    unsigned long long old=atomicAdd(counter,0ull);
    while(old<p.config.node_budget) {
        const unsigned long long found=atomicCAS(counter,old,old+1);
        if(found==old) return true;
        old=found;
    }
    pool_fail(p,CHEN_BOUNDED_ABORT_NODES);return false;
}
__global__ void pool_reset_kernel(PoolState *state,ChenBoundedJob job,uint64_t epoch,uint64_t *history) {
    if(threadIdx.x || blockIdx.x) return;
    PoolState &p=*state;p=PoolState{};p.config=job;
    auto &s=p.snapshot;s.abi_version=CHEN_ROOT_POOL_ABI;s.struct_size=sizeof(s);
    s.root_hash=job.root.hash;s.root_epoch=epoch;
    s.published.abi_version=CHEN_BOUNDED_ABI;s.published.struct_size=sizeof(s.published);
    bounded_abort(s.published,CHEN_BOUNDED_ABORT_NONE);
    if(!valid_position(job.root)) {pool_fail(p,CHEN_BOUNDED_ABORT_POSITION);return;}
    uint32_t error=0;s.root_count=legal_moves(job.root,p.actions,&error);
    if(error) {pool_fail(p,CHEN_BOUNDED_ABORT_RULES);return;}
    if(s.root_count) s.fallback=p.actions[0];
    const Node root_node{NONE,0,NONE,0,0,0,0,ChenRuleMove{},0};
    const bool terminal=!s.root_count || job.root.halfmove>=100 || insufficient(job.root) ||
        repeated(&job.root,&root_node,0,history,job.history_count);
    if(terminal) {
        s.terminal=1;s.iteration_complete=1;s.published.kind=CHEN_BOUNDED_EXACT;
        s.published.score=!s.root_count && in_check(job.root,job.root.side)?-MATE:0;
        s.published.abort_reason=0;
    }
    // Child history sees the actual parent exactly once, followed by its own
    // local child position. Parent repetition above uses ONLY original history.
    history[job.history_count]=job.root.hash;
    p.incumbent=-BOUNDED_INF;p.preferred=0;
}
__global__ void pool_begin_kernel(PoolState *state,uint32_t depth) {
    if(threadIdx.x || blockIdx.x) return;
    PoolState &p=*state;auto &s=p.snapshot;
    if(s.terminal || s.abort_reason) return;
    s.requested_depth=depth;s.iteration_complete=0;s.completed_children=0;
    s.exact_children=s.upper_children=s.paused_children=0;
    p.iteration_start_nodes=s.nodes;p.incumbent=-BOUNDED_INF;p.preferred=0;p.prime=1;
    if(s.published.pv_length) for(uint32_t i=0;i<s.root_count;++i)
        if(p.actions[i].from==s.published.pv[0].from && p.actions[i].to==s.published.pv[0].to &&
            p.actions[i].promotion==s.published.pv[0].promotion) p.preferred=i;
    for(uint32_t i=0;i<s.root_count;++i) {
        auto &c=p.children[i];c=ChenRootPoolChild{};c.canonical_index=i;c.action=p.actions[i];
        c.bound.abi_version=CHEN_BOUNDED_ABI;c.bound.struct_size=sizeof(c.bound);
        bounded_abort(c.bound,CHEN_BOUNDED_ABORT_NONE);
    }
    pool_node(p); // one entered root per finite-depth iteration; rules reused
}
__global__ void pool_epoch_kernel(PoolState *state,uint64_t allowance) {
    if(threadIdx.x || blockIdx.x) return;
    state->epoch_began=uint64_t(wall_clock64());state->epoch_ticks=allowance;
}
__global__ void pool_workers_kernel(PoolState *state,BoundedScratch *scratch,
    const uint64_t *history,uint32_t *cancel) {
    // Separate workgroups spread root tasks across CUs. This first pool control
    // intentionally keeps each DFS scalar; no divergent multi-DFS lane mapping.
    if(threadIdx.x) return;
    PoolState &p=*state;auto &s=p.snapshot;
    const uint32_t slot=blockIdx.x;
    if(slot>=s.root_count || s.terminal || s.iteration_complete || (p.prime && slot)) return;
    // Rotate previous PV to workgroup0, preserving canonical action identity.
    const uint32_t index=slot==0?p.preferred:slot==p.preferred?0:slot;
    auto &child=p.children[index];auto &out=child.bound;BoundedScratch &work=scratch[index];
    if(child.status==CHEN_ROOT_POOL_COMPLETE || child.status==CHEN_ROOT_POOL_FAILED) return;
    uint32_t sp=child.stack_pointer,transitions=0;
    if(child.status==CHEN_ROOT_POOL_PENDING) {
        const int32_t incumbent=__hip_atomic_load(&p.incumbent,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT);
        child.original_alpha=-BOUNDED_INF;
        child.original_beta=incumbent==-BOUNDED_INF?BOUNDED_INF:1-incumbent;
        work.rules_error=0;work.positions[0]=apply<true>(p.config.root,child.action);
        work.nodes[0]=Node{NONE,0,NONE,0,1,0,0,child.action,0};
        work.frames[0]=BoundedFrame{int32_t(s.requested_depth)-1,child.original_alpha,child.original_beta,-BOUNDED_INF,0,0,0,0};
        child.status=CHEN_ROOT_POOL_PAUSED;
    }
    while(true) {
        if((transitions++&7)==0) {
            child.stack_pointer=sp;
            if(__hip_atomic_load(cancel,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)) {
                pool_fail(p,CHEN_BOUNDED_ABORT_CANCELLED);return;
            }
            if(__hip_atomic_load(&s.abort_reason,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)) return;
            if(uint64_t(wall_clock64())-p.epoch_began>=p.epoch_ticks) return;
        }
        auto &frame=work.frames[sp];const auto &position=work.positions[sp];
        bool complete=false;
        if(!frame.entered) {
            if(!pool_node(p)) {child.status=CHEN_ROOT_POOL_FAILED;bounded_abort(out,CHEN_BOUNDED_ABORT_NODES);return;}
            ++out.nodes;const bool qsearch=p.config.quiescence && frame.depth<=0;
            if(qsearch) {++out.qnodes;atomicAdd(reinterpret_cast<unsigned long long *>(&s.qnodes),1ull);}
            frame.entered=1;work.pv_length[sp]=0;
            frame.count=legal_moves(position,work.moves[sp],&work.rules_error);
            if(work.rules_error) {child.status=CHEN_ROOT_POOL_FAILED;bounded_abort(out,CHEN_BOUNDED_ABORT_RULES);pool_fail(p,CHEN_BOUNDED_ABORT_RULES);return;}
            const bool check=in_check(position,position.side);
            if(!frame.count) {frame.best=check?-MATE+int(1+sp):0;complete=true;}
            else if(position.halfmove>=100 || insufficient(position) ||
                repeated(work.positions,work.nodes,sp,history,p.config.history_count+1)) {frame.best=0;complete=true;}
            else if(frame.depth<=0 && !qsearch) {frame.best=evaluate(position,p.config.evaluator);complete=true;}
            else if(qsearch) {
                frame.best=check?-BOUNDED_INF:evaluate(position,p.config.evaluator);
                if(!check) {
                    if(frame.best>=frame.beta) complete=true;
                    else {
                        if(frame.best>frame.alpha) frame.alpha=frame.best;
                        uint32_t selected=0;for(uint32_t i=0;i<frame.count;++i)
                            if(bounded_tactical(position,work.moves[sp][i])) work.moves[sp][selected++]=work.moves[sp][i];
                        frame.count=selected;
                    }
                }
                if(!complete && frame.count && frame.qply>=p.config.max_qply) {
                    child.status=CHEN_ROOT_POOL_FAILED;bounded_abort(out,CHEN_BOUNDED_ABORT_QPLY);pool_fail(p,CHEN_BOUNDED_ABORT_QPLY);return;
                }
            }
        }
        if(!complete && (frame.alpha>=frame.beta || frame.cursor==frame.count)) complete=true;
        if(complete) {
            const int32_t value=frame.best;
            if(!sp) {
                // No bound publication on cancellation/elapsed failure. Resume
                // completed frame later without re-entering or re-counting it.
                child.stack_pointer=0;
                if(__hip_atomic_load(cancel,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)) {pool_fail(p,CHEN_BOUNDED_ABORT_CANCELLED);return;}
                if(uint64_t(wall_clock64())-p.epoch_began>=p.epoch_ticks) {frame.cursor=frame.count;frame.entered=2;return;}
                out.score=value;out.completed_depth=s.requested_depth-1;out.abort_reason=0;
                out.kind=value<=child.original_alpha?CHEN_BOUNDED_UPPER:value>=child.original_beta?CHEN_BOUNDED_LOWER:CHEN_BOUNDED_EXACT;
                out.pv_length=work.pv_length[0];for(uint32_t i=0;i<out.pv_length;++i) out.pv[i]=work.pv[0][i];
                child.status=CHEN_ROOT_POOL_COMPLETE;
                if(out.kind==CHEN_BOUNDED_EXACT) __hip_atomic_fetch_max(&p.incumbent,-value,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
                return;
            }
            const uint32_t below=sp--;auto &parent=work.frames[sp];const int32_t score=-value;
            if(score>parent.best) {
                parent.best=score;const uint32_t length=work.pv_length[below];
                if(length+1>=CHEN_BOUNDED_STACK) {child.status=CHEN_ROOT_POOL_FAILED;bounded_abort(out,CHEN_BOUNDED_ABORT_STACK);pool_fail(p,CHEN_BOUNDED_ABORT_STACK);return;}
                work.pv[sp][0]=work.moves[sp][parent.cursor-1];
                for(uint32_t i=0;i<length;++i) work.pv[sp][i+1]=work.pv[below][i];work.pv_length[sp]=length+1;
            }
            if(score>parent.alpha) parent.alpha=score;
            continue;
        }
        if(sp+2>=CHEN_BOUNDED_STACK) {child.status=CHEN_ROOT_POOL_FAILED;bounded_abort(out,CHEN_BOUNDED_ABORT_STACK);pool_fail(p,CHEN_BOUNDED_ABORT_STACK);return;}
        const auto move=work.moves[sp][frame.cursor++];const bool qsearch=p.config.quiescence && frame.depth<=0;
        work.positions[sp+1]=apply<true>(position,move);
        work.nodes[sp+1]=Node{sp,0,NONE,0,sp+2,0,0,move,0};
        work.frames[sp+1]=BoundedFrame{qsearch?0:frame.depth-1,-frame.beta,-frame.alpha,-BOUNDED_INF,qsearch?frame.qply+1:0,0,0,0};++sp;
    }
}
__global__ void pool_reduce_kernel(PoolState *state,uint32_t *cancel) {
    if(threadIdx.x || blockIdx.x) return;
    PoolState &p=*state;auto &s=p.snapshot;p.prime=0;s.iteration_nodes=s.nodes-p.iteration_start_nodes;
    s.elapsed_ticks=uint64_t(wall_clock64())-p.epoch_began;
    s.completed_children=s.exact_children=s.upper_children=s.paused_children=0;
    int32_t best=-BOUNDED_INF;uint32_t winner=UINT32_MAX;
    for(uint32_t i=0;i<s.root_count;++i) {
        const auto &c=p.children[i];
        if(c.status==CHEN_ROOT_POOL_COMPLETE) {
            ++s.completed_children;
            if(c.bound.kind==CHEN_BOUNDED_EXACT) {++s.exact_children;if(-c.bound.score>best) {best=-c.bound.score;winner=i;}}
            if(c.bound.kind==CHEN_BOUNDED_LOWER) ++s.upper_children;
        } else if(c.status==CHEN_ROOT_POOL_PAUSED) ++s.paused_children;
    }
    if(__hip_atomic_load(cancel,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)) pool_fail(p,CHEN_BOUNDED_ABORT_CANCELLED);
    if(s.abort_reason || winner==UINT32_MAX) return;
    for(uint32_t i=0;i<s.root_count;++i) if(i!=winner) {
        const auto &c=p.children[i];if(c.status!=CHEN_ROOT_POOL_COMPLETE) return;
        if(c.bound.kind!=CHEN_BOUNDED_EXACT && c.bound.kind!=CHEN_BOUNDED_LOWER) return;
        const int32_t upper=-c.bound.score;
        if(upper>best || (i<winner && upper==best)) return;
    }
    auto &out=s.published;out=ChenBoundedResult{};out.abi_version=1;out.struct_size=sizeof(out);out.kind=CHEN_BOUNDED_EXACT;
    out.score=best;out.completed_depth=s.requested_depth;out.nodes=s.iteration_nodes;out.qnodes=s.qnodes;
    const auto &child=p.children[winner];out.pv_length=child.bound.pv_length+1;out.pv[0]=child.action;
    for(uint32_t i=0;i<child.bound.pv_length;++i) out.pv[i+1]=child.bound.pv[i];s.iteration_complete=1;
}
}
struct ChenRootPoolContext {
    int32_t device;bool poisoned,ready,completed;uint32_t depth;
    hipStream_t stream;hipEvent_t events[4];PoolState *state;BoundedScratch *scratch;
    uint64_t *history;uint64_t clock_khz;uint32_t *host_cancel,*device_cancel;
};
namespace {
int pool_error(ChenRootPoolContext *c,hipError_t code,char *error,uint64_t cap) {
    if(c && c->stream && hipStreamSynchronize(c->stream)!=hipSuccess) {c->poisoned=true;return bounded_failure(error,cap,"pool retirement failed; retire caller process");}
    return bounded_failure(error,cap,hipGetErrorString(code));
}
int pool_snapshot(ChenRootPoolContext *c,ChenRootPoolSnapshot *snapshot,char *error,uint64_t cap) {
    ChenRootPoolSnapshot staged{};hipError_t code=hipMemcpyAsync(&staged,&c->state->snapshot,sizeof(staged),hipMemcpyDeviceToHost,c->stream);
    if(code!=hipSuccess) return pool_error(c,code,error,cap);
    if(hipStreamSynchronize(c->stream)!=hipSuccess) {c->poisoned=true;return bounded_failure(error,cap,"pool retirement failed; retire caller process");}
    *snapshot=staged;return 0;
}
}
extern "C" uint32_t chen_root_pool_abi_version() {return CHEN_ROOT_POOL_ABI;}
extern "C" void chen_root_pool_cancel(ChenRootPoolContext *c) {if(c && c->host_cancel)__atomic_store_n(c->host_cancel,1u,__ATOMIC_SEQ_CST);}
extern "C" int32_t chen_root_pool_destroy(ChenRootPoolContext *c) {
    if(!c)return 0;if(c->poisoned)return -1;hipError_t code=hipSetDevice(c->device);
    if(code!=hipSuccess || hipStreamSynchronize(c->stream)!=hipSuccess){c->poisoned=true;return -1;}
    hipError_t first=hipSuccess;
    for(void *p:{static_cast<void *>(c->state),static_cast<void *>(c->scratch),static_cast<void *>(c->history)})if(p){code=hipFree(p);if(first==hipSuccess)first=code;}
    if(c->host_cancel){code=hipHostFree(c->host_cancel);if(first==hipSuccess)first=code;}
    for(auto e:c->events)if(e){code=hipEventDestroy(e);if(first==hipSuccess)first=code;}
    if(c->stream){code=hipStreamDestroy(c->stream);if(first==hipSuccess)first=code;}delete c;return first==hipSuccess?0:-1;
}
extern "C" ChenRootPoolContext *chen_root_pool_create(int32_t device,char *error,uint64_t cap) {
    if(error && cap)error[0]=0;
    if(std::getenv("HSA_OVERRIDE_GFX_VERSION")){bounded_failure(error,cap,"architecture override forbidden");return nullptr;}
    hipDeviceProp_t prop{};int khz=0;hipError_t code=hipSetDevice(device);
    if(code==hipSuccess)code=hipGetDeviceProperties(&prop,device);
    if(code==hipSuccess)code=hipDeviceGetAttribute(&khz,hipDeviceAttributeWallClockRate,device);
    if(code!=hipSuccess || khz<=0 || khz>1000000000 || !prop.canMapHostMemory || std::strncmp(prop.gcnArchName,"gfx1030",7) || (prop.gcnArchName[7] && prop.gcnArchName[7]!=':')){bounded_failure(error,cap,"pool requires actual gfx1030 and coherent host mapping/clock");return nullptr;}
    auto *c=new(std::nothrow)ChenRootPoolContext{};if(!c){bounded_failure(error,cap,"pool host allocation failed");return nullptr;}
    c->device=device;c->clock_khz=uint64_t(khz);
    code=hipStreamCreateWithFlags(&c->stream,hipStreamNonBlocking);
    for(int i=0;i<4 && code==hipSuccess;++i)code=hipEventCreate(&c->events[i]);
    if(code==hipSuccess)code=hipMalloc(&c->state,sizeof(PoolState));
    if(code==hipSuccess)code=hipMalloc(&c->scratch,CHEN_RULES_MAX_MOVES*sizeof(BoundedScratch));
    if(code==hipSuccess)code=hipMalloc(&c->history,65536*sizeof(uint64_t));
    if(code==hipSuccess)code=hipHostMalloc(&c->host_cancel,sizeof(uint32_t),hipHostMallocMapped|hipHostMallocCoherent);
    void *mapped=nullptr;if(code==hipSuccess)code=hipHostGetDevicePointer(&mapped,c->host_cancel,0);c->device_cancel=static_cast<uint32_t *>(mapped);
    if(code!=hipSuccess){pool_error(c,code,error,cap);if(chen_root_pool_destroy(c))bounded_failure(error,cap,"pool creation cleanup failed; retire process");return nullptr;}
    __atomic_store_n(c->host_cancel,0u,__ATOMIC_SEQ_CST);return c;
}
extern "C" int32_t chen_root_pool_reset(ChenRootPoolContext *c,const ChenBoundedJob *job,const uint64_t *history,uint64_t count,uint64_t epoch,ChenRootPoolSnapshot *snapshot,char *error,uint64_t cap) {
    if(error && cap)error[0]=0;
    if(!c || c->poisoned || !job || !snapshot || count>65535 || (count && !history) || job->abi_version!=1 || job->struct_size!=sizeof(*job) || job->depth || job->root_ply || job->history_offset || job->history_count!=count || job->reserved || job->max_qply>32 || job->evaluator>1 || job->quiescence>1 || job->alpha!=-BOUNDED_INF || job->beta!=BOUNDED_INF || job->node_budget>CHEN_BOUNDED_MAX_NODES)return bounded_failure(error,cap,"invalid pool reset");
    hipError_t code=hipSetDevice(c->device);__atomic_store_n(c->host_cancel,0u,__ATOMIC_SEQ_CST);
    if(code==hipSuccess && count)code=hipMemcpyAsync(c->history,history,count*8,hipMemcpyHostToDevice,c->stream);
    if(code==hipSuccess){hipLaunchKernelGGL(pool_reset_kernel,dim3(1),dim3(64),0,c->stream,c->state,*job,epoch,c->history);code=hipGetLastError();}
    if(code!=hipSuccess)return pool_error(c,code,error,cap);
    if(pool_snapshot(c,snapshot,error,cap))return -1;c->ready=true;c->depth=0;c->completed=true;return 0;
}
extern "C" int32_t chen_root_pool_begin(ChenRootPoolContext *c,uint32_t depth,ChenRootPoolSnapshot *snapshot,char *error,uint64_t cap) {
    if(error && cap)error[0]=0;
    if(!c || c->poisoned || !c->ready || !snapshot || !depth || depth>16 || depth!=c->depth+1 || !c->completed)return bounded_failure(error,cap,"invalid pool depth transition");
    hipError_t code=hipSetDevice(c->device);
    if(code==hipSuccess){hipLaunchKernelGGL(pool_begin_kernel,dim3(1),dim3(64),0,c->stream,c->state,depth);code=hipGetLastError();}
    if(code!=hipSuccess)return pool_error(c,code,error,cap);
    if(pool_snapshot(c,snapshot,error,cap))return -1;c->depth=depth;c->completed=false;return 0;
}
extern "C" int32_t chen_root_pool_advance(ChenRootPoolContext *c,uint64_t ns,ChenRootPoolSnapshot *snapshot,ChenRootPoolChild *children,uint32_t capacity,ChenBoundedTimedTelemetry *telemetry,char *error,uint64_t cap) {
    if(error && cap)error[0]=0;
    if(!c || c->poisoned || !c->ready || !c->depth || !snapshot || !telemetry || !ns || ns>50000000 || (children?capacity!=256:capacity!=0))return bounded_failure(error,cap,"invalid pool epoch buffers/allowance");
    ChenRootPoolSnapshot staged{};ChenRootPoolChild staged_children[256]{};ChenBoundedTimedTelemetry stats{};
    hipError_t code=hipSetDevice(c->device);
    if(code==hipSuccess)code=hipEventRecord(c->events[0],c->stream);
    if(code==hipSuccess){hipLaunchKernelGGL(pool_epoch_kernel,dim3(1),dim3(64),0,c->stream,c->state,(ns*c->clock_khz+999999)/1000000);code=hipGetLastError();}
    if(code==hipSuccess){hipLaunchKernelGGL(pool_workers_kernel,dim3(256),dim3(64),0,c->stream,c->state,c->scratch,c->history,c->device_cancel);code=hipGetLastError();}
    if(code==hipSuccess){hipLaunchKernelGGL(pool_reduce_kernel,dim3(1),dim3(64),0,c->stream,c->state,c->device_cancel);code=hipGetLastError();}
    if(code==hipSuccess)code=hipEventRecord(c->events[1],c->stream);
    if(code==hipSuccess)code=hipMemcpyAsync(&staged,&c->state->snapshot,sizeof(staged),hipMemcpyDeviceToHost,c->stream);
    if(code==hipSuccess && children)code=hipMemcpyAsync(staged_children,&c->state->children,sizeof(staged_children),hipMemcpyDeviceToHost,c->stream);
    if(code==hipSuccess)code=hipEventRecord(c->events[2],c->stream);
    if(code!=hipSuccess)return pool_error(c,code,error,cap);
    if(hipStreamSynchronize(c->stream)!=hipSuccess){c->poisoned=true;return bounded_failure(error,cap,"pool retirement failed; retire caller process");}
    float elapsed=0;code=hipEventElapsedTime(&elapsed,c->events[0],c->events[1]);stats.kernel_ms=elapsed;
    if(code==hipSuccess)code=hipEventElapsedTime(&elapsed,c->events[1],c->events[2]);stats.d2h_ms=elapsed;
    if(code!=hipSuccess)return pool_error(c,code,error,cap);
    stats.wall_clock_khz=c->clock_khz;stats.scratch_bytes=256*sizeof(BoundedScratch);stats.history_capacity=65536;
    c->completed=staged.iteration_complete && !staged.abort_reason;*snapshot=staged;*telemetry=stats;if(children)std::memcpy(children,staged_children,sizeof(staged_children));return 0;
}
