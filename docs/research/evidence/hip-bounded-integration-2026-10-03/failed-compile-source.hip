// Isolated ABI2 integration: legacy ABI1 search source is unchanged.
#include "bounded_search.hip"
#include "bounded_search_timed.h"
namespace {
__global__ void bounded_timed_kernel(const ChenBoundedTimedJob *jobs,uint32_t count,
    const uint64_t *history,BoundedScratch *scratch,ChenBoundedTimedResult *results,
    uint32_t *cancel,uint64_t clock_khz) {
    const uint32_t job_index=blockIdx.x*blockDim.x+threadIdx.x;
    if(job_index>=count) return;
    const ChenBoundedTimedJob timed_job=jobs[job_index];
    const ChenBoundedJob job=timed_job.search;
    ChenBoundedTimedResult &packet=results[job_index];
    packet=ChenBoundedTimedResult{};packet.abi_version=CHEN_BOUNDED_TIMED_ABI;
    packet.struct_size=sizeof(packet);
    const uint64_t began=uint64_t(wall_clock64());
    // ceil(ns * kHz / 1e6), admitted operands cannot overflow uint64.
    const uint64_t allowance=(timed_job.elapsed_ns*clock_khz+999999)/1000000;
    uint32_t transitions=0;
    ChenBoundedResult &out=packet.search;
    out=ChenBoundedResult{};out.abi_version=CHEN_BOUNDED_ABI;out.struct_size=sizeof(out);
    bounded_abort(out,CHEN_BOUNDED_ABORT_NONE);
    if(!valid_position(job.root)) { bounded_abort(out,CHEN_BOUNDED_ABORT_POSITION);packet.elapsed_ticks=uint64_t(wall_clock64())-began;return; }
    BoundedScratch &work=scratch[job_index];
    uint32_t sp=0;work.rules_error=0;
    work.positions[0]=job.root;
    work.nodes[0]=Node{NONE,0,NONE,0,job.root_ply,0,0,ChenRuleMove{},0};
    work.frames[0]=BoundedFrame{int32_t(job.depth),job.alpha,job.beta,-BOUNDED_INF,0,0,0,0};
    while(true) {
        if((transitions++&31)==0) {
            packet.elapsed_ticks=uint64_t(wall_clock64())-began;
            if(__hip_atomic_load(cancel,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)) {
                bounded_abort(out,CHEN_BOUNDED_ABORT_CANCELLED);return;
            }
            if(packet.elapsed_ticks>=allowance) {
                bounded_abort(out,CHEN_BOUNDED_ABORT_DEADLINE);return;
            }
        }
        BoundedFrame &frame=work.frames[sp];
        const ChenRulePosition &position=work.positions[sp];
        bool complete=false;
        if(!frame.entered) {
            if(out.nodes>=job.node_budget) { bounded_abort(out,CHEN_BOUNDED_ABORT_NODES);packet.elapsed_ticks=uint64_t(wall_clock64())-began;return; }
            ++out.nodes;
            const bool qsearch=job.quiescence && frame.depth<=0;
            if(qsearch) ++out.qnodes;
            frame.entered=1;work.pv_length[sp]=0;
            frame.count=legal_moves(position,work.moves[sp],&work.rules_error);
            if(!sp && !work.rules_error) {
                packet.legal_root_count=frame.count;
                if(frame.count) packet.fallback=work.moves[sp][0];
            }
            if(work.rules_error) { bounded_abort(out,CHEN_BOUNDED_ABORT_RULES);packet.elapsed_ticks=uint64_t(wall_clock64())-began;return; }
            const bool check=in_check(position,position.side);
            // Mate/stalemate has precedence over draw counters, as in both
            // existing rule classifiers and the CPU baseline.
            if(!frame.count) {
                frame.best=check?-MATE+int(job.root_ply+sp):0;complete=true;
            } else if(position.halfmove>=100 || insufficient(position) ||
                repeated(work.positions,work.nodes,sp,
                    history?history+job.history_offset:nullptr,job.history_count)) {
                frame.best=0;complete=true;
            } else if(frame.depth<=0 && !qsearch) {
                frame.best=evaluate(position,job.evaluator);complete=true;
            } else if(qsearch) {
                frame.best=check?-BOUNDED_INF:evaluate(position,job.evaluator);
                if(!check) {
                    if(frame.best>=frame.beta) complete=true;
                    else {
                        if(frame.best>frame.alpha) frame.alpha=frame.best;
                        uint32_t selected=0;
                        for(uint32_t i=0;i<frame.count;++i)
                            if(bounded_tactical(position,work.moves[sp][i]))
                                work.moves[sp][selected++]=work.moves[sp][i];
                        frame.count=selected;
                    }
                }
                // Empty tactical set is closed. In check, every legal evasion
                // remains; never quietly substitute static evaluation at a cap.
                if(!complete && frame.count && frame.qply>=job.max_qply) {
                    bounded_abort(out,CHEN_BOUNDED_ABORT_QPLY);packet.elapsed_ticks=uint64_t(wall_clock64())-began;return;
                }
            }
        }
        if(!complete && (frame.alpha>=frame.beta || frame.cursor==frame.count)) complete=true;
        if(complete) {
            const int32_t value=frame.best;
            if(!sp) {
                packet.elapsed_ticks=uint64_t(wall_clock64())-began;
                if(__hip_atomic_load(cancel,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)) {
                    bounded_abort(out,CHEN_BOUNDED_ABORT_CANCELLED);return;
                }
                if(packet.elapsed_ticks>=allowance) {
                    bounded_abort(out,CHEN_BOUNDED_ABORT_DEADLINE);return;
                }
                out.score=value;out.completed_depth=job.depth;out.abort_reason=CHEN_BOUNDED_ABORT_NONE;
                // Classify against the ORIGINAL job window, never mutated alpha.
                out.kind=value<=job.alpha?CHEN_BOUNDED_UPPER:
                    value>=job.beta?CHEN_BOUNDED_LOWER:CHEN_BOUNDED_EXACT;
                out.pv_length=work.pv_length[0];
                for(uint32_t i=0;i<out.pv_length;++i) out.pv[i]=work.pv[0][i];
                return;
            }
            const uint32_t child=sp--;
            BoundedFrame &parent=work.frames[sp];
            const int32_t score=-value;
            if(score>parent.best) {
                parent.best=score;
                const uint32_t length=work.pv_length[child];
                if(length+1>CHEN_BOUNDED_STACK) { bounded_abort(out,CHEN_BOUNDED_ABORT_STACK);packet.elapsed_ticks=uint64_t(wall_clock64())-began;return; }
                work.pv[sp][0]=work.moves[sp][parent.cursor-1];
                for(uint32_t i=0;i<length;++i) work.pv[sp][i+1]=work.pv[child][i];
                work.pv_length[sp]=length+1;
            }
            if(score>parent.alpha) parent.alpha=score;
            continue;
        }
        if(sp+1>=CHEN_BOUNDED_STACK) { bounded_abort(out,CHEN_BOUNDED_ABORT_STACK);packet.elapsed_ticks=uint64_t(wall_clock64())-began;return; }
        const ChenRuleMove move=work.moves[sp][frame.cursor++];
        const bool qsearch=job.quiescence && frame.depth<=0;
        work.positions[sp+1]=apply<true>(position,move);
        work.nodes[sp+1]=Node{sp,0,NONE,0,job.root_ply+sp+1,0,0,move,0};
        work.frames[sp+1]=BoundedFrame{qsearch?0:frame.depth-1,-frame.beta,-frame.alpha,
            -BOUNDED_INF,qsearch?frame.qply+1:0,0,0,0};
        ++sp;
    }
}
}
struct ChenBoundedTimedContext {
    int32_t device;uint32_t capacity;bool poisoned;
    uint64_t history_count,clock_khz;
    hipStream_t stream;
    hipEvent_t events[4];
    ChenBoundedTimedJob *jobs;
    ChenBoundedTimedResult *results;
    BoundedScratch *scratch;
    uint64_t *history;
    uint32_t *host_cancel,*device_cancel;
};
namespace {
int32_t timed_error(ChenBoundedTimedContext *ctx,hipError_t code,char *error,uint64_t cap) {
    if(ctx && ctx->stream) {
        const hipError_t retired=hipStreamSynchronize(ctx->stream);
        if(retired!=hipSuccess) {
            ctx->poisoned=true;
            return bounded_failure(error,cap,"HIP retirement failed; retire caller process");
        }
    }
    return bounded_failure(error,cap,hipGetErrorString(code));
}
bool timed_job_valid(const ChenBoundedTimedJob &timed,uint64_t history_count) {
    const ChenBoundedJob &job=timed.search;
    return timed.abi_version==CHEN_BOUNDED_TIMED_ABI && timed.struct_size==sizeof(timed) &&
        timed.elapsed_ns && timed.elapsed_ns<=CHEN_BOUNDED_TIMED_MAX_NS &&
        job.abi_version==CHEN_BOUNDED_ABI && job.struct_size==sizeof(job) && !job.reserved &&
        job.depth<=CHEN_BOUNDED_MAX_DEPTH && job.max_qply<=CHEN_BOUNDED_MAX_QPLY &&
        job.root_ply<CHEN_BOUNDED_STACK && job.depth+job.max_qply+job.root_ply<CHEN_BOUNDED_STACK &&
        job.evaluator<=1 && job.quiescence<=1 && job.node_budget<=CHEN_BOUNDED_MAX_NODES &&
        job.alpha>=-BOUNDED_INF && job.alpha<job.beta && job.beta<=BOUNDED_INF &&
        job.history_offset<=history_count && job.history_count<=history_count-job.history_offset;
}
}
extern "C" uint32_t chen_bounded_timed_abi_version() { return CHEN_BOUNDED_TIMED_ABI; }
extern "C" void chen_bounded_timed_cancel(ChenBoundedTimedContext *ctx) {
    if(ctx && ctx->host_cancel) __atomic_store_n(ctx->host_cancel,1u,__ATOMIC_SEQ_CST);
}
extern "C" void chen_bounded_timed_clear_cancel(ChenBoundedTimedContext *ctx) {
    if(ctx && ctx->host_cancel) __atomic_store_n(ctx->host_cancel,0u,__ATOMIC_SEQ_CST);
}
extern "C" int32_t chen_bounded_timed_destroy(ChenBoundedTimedContext *ctx) {
    if(!ctx) return 0;
    if(ctx->poisoned) return -1;
    hipError_t status=hipSetDevice(ctx->device);
    if(status!=hipSuccess) return -1;
    if(ctx->stream && hipStreamSynchronize(ctx->stream)!=hipSuccess) { ctx->poisoned=true;return -1; }
    hipError_t first=hipSuccess;
    void *buffers[]={ctx->jobs,ctx->results,ctx->scratch,ctx->history};
    for(void *buffer:buffers) if(buffer) {
        status=hipFree(buffer);if(first==hipSuccess) first=status;
    }
    if(ctx->host_cancel) { status=hipHostFree(ctx->host_cancel);if(first==hipSuccess) first=status; }
    for(hipEvent_t event:ctx->events) if(event) {
        status=hipEventDestroy(event);if(first==hipSuccess) first=status;
    }
    if(ctx->stream) { status=hipStreamDestroy(ctx->stream);if(first==hipSuccess) first=status; }
    delete ctx;return first==hipSuccess?0:-1;
}
extern "C" ChenBoundedTimedContext *chen_bounded_timed_create(int32_t device,uint32_t capacity,
    char *error,uint64_t cap) {
    if(error && cap) error[0]=0;
    if(!capacity || capacity>CHEN_BOUNDED_MAX_JOBS || std::getenv("HSA_OVERRIDE_GFX_VERSION")) {
        bounded_failure(error,cap,"invalid timed capacity or architecture override");return nullptr;
    }
    hipError_t code=hipSetDevice(device);
    hipDeviceProp_t properties{};
    if(code==hipSuccess) code=hipGetDeviceProperties(&properties,device);
    if(code!=hipSuccess) { bounded_failure(error,cap,hipGetErrorString(code));return nullptr; }
    int clock_khz=0;
    code=hipDeviceGetAttribute(&clock_khz,hipDeviceAttributeWallClockRate,device);
    if(code!=hipSuccess || clock_khz<=0 || clock_khz>1000000000 || !properties.canMapHostMemory ||
        std::strncmp(properties.gcnArchName,"gfx1030",7) ||
        (properties.gcnArchName[7] && properties.gcnArchName[7]!=':')) {
        bounded_failure(error,cap,"timed reference requires gfx1030, positive wall clock and mapped host memory");return nullptr;
    }
    auto *ctx=new(std::nothrow) ChenBoundedTimedContext{};
    if(!ctx) { bounded_failure(error,cap,"cannot allocate timed context");return nullptr; }
    ctx->device=device;ctx->capacity=capacity;ctx->clock_khz=uint64_t(clock_khz);
    code=hipStreamCreateWithFlags(&ctx->stream,hipStreamNonBlocking);
    for(int i=0;i<4 && code==hipSuccess;++i) code=hipEventCreate(&ctx->events[i]);
    if(code==hipSuccess) code=hipMalloc(&ctx->jobs,size_t(capacity)*sizeof(*ctx->jobs));
    if(code==hipSuccess) code=hipMalloc(&ctx->results,size_t(capacity)*sizeof(*ctx->results));
    if(code==hipSuccess) code=hipMalloc(&ctx->scratch,size_t(capacity)*sizeof(*ctx->scratch));
    if(code==hipSuccess) code=hipMalloc(&ctx->history,65535*sizeof(uint64_t));
    if(code==hipSuccess) code=hipHostMalloc(&ctx->host_cancel,sizeof(uint32_t),hipHostMallocMapped|hipHostMallocCoherent);
    if(code==hipSuccess) code=hipHostGetDevicePointer(&ctx->device_cancel,ctx->host_cancel,0);
    if(code!=hipSuccess) {
        timed_error(ctx,code,error,cap);
        if(chen_bounded_timed_destroy(ctx)) std::snprintf(error,size_t(cap),"timed creation cleanup failed; retire caller process");
        return nullptr;
    }
    chen_bounded_timed_clear_cancel(ctx);return ctx;
}
extern "C" int32_t chen_bounded_timed_history(ChenBoundedTimedContext *ctx,const uint64_t *history,
    uint64_t count,double *h2d_ms,char *error,uint64_t cap) {
    if(error && cap) error[0]=0;
    if(!ctx || ctx->poisoned || count>65535 || (count && !history) || !h2d_ms)
        return bounded_failure(error,cap,"invalid timed history/context");
    hipError_t code=hipSetDevice(ctx->device);
    if(code==hipSuccess) code=hipEventRecord(ctx->events[0],ctx->stream);
    if(code==hipSuccess && count) code=hipMemcpyAsync(ctx->history,history,size_t(count)*8,hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->events[1],ctx->stream);
    if(code!=hipSuccess) return timed_error(ctx,code,error,cap);
    code=hipStreamSynchronize(ctx->stream);
    if(code!=hipSuccess) { ctx->poisoned=true;return bounded_failure(error,cap,"HIP retirement failed; retire caller process"); }
    float elapsed=0;code=hipEventElapsedTime(&elapsed,ctx->events[0],ctx->events[1]);
    if(code!=hipSuccess) return timed_error(ctx,code,error,cap);
    ctx->history_count=count;*h2d_ms=elapsed;return 0;
}
extern "C" int32_t chen_bounded_timed_run(ChenBoundedTimedContext *ctx,
    const ChenBoundedTimedJob *jobs,uint32_t count,ChenBoundedTimedResult *results,
    ChenBoundedTimedTelemetry *telemetry,char *error,uint64_t cap) {
    if(error && cap) error[0]=0;
    if(!ctx || ctx->poisoned || !jobs || !results || !telemetry || !count || count>ctx->capacity)
        return bounded_failure(error,cap,"invalid timed run buffers/context/count");
    for(uint32_t i=0;i<count;++i) if(!timed_job_valid(jobs[i],ctx->history_count))
        return bounded_failure(error,cap,"invalid timed job configuration");
    ChenBoundedTimedResult staged[CHEN_BOUNDED_MAX_JOBS]{};
    hipError_t code=hipSetDevice(ctx->device);
    if(code==hipSuccess) code=hipEventRecord(ctx->events[0],ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(ctx->jobs,jobs,size_t(count)*sizeof(*jobs),hipMemcpyHostToDevice,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->events[1],ctx->stream);
    if(code==hipSuccess) {
        hipLaunchKernelGGL(bounded_timed_kernel,dim3((count+63)/64),dim3(64),0,ctx->stream,
            ctx->jobs,count,ctx->history,ctx->scratch,ctx->results,ctx->device_cancel,ctx->clock_khz);
        code=hipGetLastError();
    }
    if(code==hipSuccess) code=hipEventRecord(ctx->events[2],ctx->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(staged,ctx->results,size_t(count)*sizeof(*results),hipMemcpyDeviceToHost,ctx->stream);
    if(code==hipSuccess) code=hipEventRecord(ctx->events[3],ctx->stream);
    if(code!=hipSuccess) return timed_error(ctx,code,error,cap);
    code=hipStreamSynchronize(ctx->stream);
    if(code!=hipSuccess) { ctx->poisoned=true;return bounded_failure(error,cap,"HIP retirement failed; retire caller process"); }
    float times[3]{};
    for(int i=0;i<3 && code==hipSuccess;++i) code=hipEventElapsedTime(&times[i],ctx->events[i],ctx->events[i+1]);
    if(code!=hipSuccess) return timed_error(ctx,code,error,cap);
    const ChenBoundedTimedTelemetry measured{times[0],times[1],times[2],ctx->clock_khz,
        uint64_t(ctx->capacity)*sizeof(BoundedScratch),65535};
    std::memcpy(results,staged,size_t(count)*sizeof(*results));*telemetry=measured;return 0;
}
