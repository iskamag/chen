// Isolated research translation unit: reuses byte-identical private production
// rules, evaluation and history helpers. No shared production source changes.
#include "../chen_rules.hip"
#include "bounded_search.h"

namespace {
constexpr int BOUNDED_INF=32000;
struct BoundedFrame {
    int32_t depth,alpha,beta,best;
    uint32_t qply,cursor,count,entered;
};
struct BoundedScratch {
    ChenRulePosition positions[CHEN_BOUNDED_STACK];
    Node nodes[CHEN_BOUNDED_STACK];
    BoundedFrame frames[CHEN_BOUNDED_STACK];
    ChenRuleMove moves[CHEN_BOUNDED_STACK][CHEN_RULES_MAX_MOVES];
    ChenRuleMove pv[CHEN_BOUNDED_STACK][CHEN_BOUNDED_STACK];
    uint32_t pv_length[CHEN_BOUNDED_STACK];
    uint32_t rules_error,reserved;
};
static_assert(sizeof(BoundedScratch)==CHEN_BOUNDED_STACK*(sizeof(ChenRulePosition)+sizeof(Node)+
    sizeof(BoundedFrame)+CHEN_RULES_MAX_MOVES*sizeof(ChenRuleMove)+
    CHEN_BOUNDED_STACK*sizeof(ChenRuleMove)+sizeof(uint32_t))+8,"bounded scratch shape");
__device__ bool bounded_tactical(const ChenRulePosition &p,ChenRuleMove move) {
    if(move.promotion || piece_at(p,move.to)>=0) return true;
    const int piece=piece_at(p,move.from);
    return piece>=0 && piece%6==0 && move.from%8!=move.to%8;
}
__device__ void bounded_abort(ChenBoundedResult &out,uint32_t reason) {
    out.kind=CHEN_BOUNDED_ABORTED;out.abort_reason=reason;
    out.score=0;out.completed_depth=0;out.pv_length=0;
    for(uint32_t i=0;i<CHEN_BOUNDED_STACK;++i) out.pv[i]=ChenRuleMove{};
}
__global__ void bounded_search_kernel(const ChenBoundedJob *jobs,uint32_t count,
    const uint64_t *history,BoundedScratch *scratch,ChenBoundedResult *results) {
    const uint32_t job_index=blockIdx.x*blockDim.x+threadIdx.x;
    if(job_index>=count) return;
    const ChenBoundedJob job=jobs[job_index];
    ChenBoundedResult &out=results[job_index];
    out=ChenBoundedResult{};out.abi_version=CHEN_BOUNDED_ABI;out.struct_size=sizeof(out);
    bounded_abort(out,CHEN_BOUNDED_ABORT_NONE);
    if(!valid_position(job.root)) { bounded_abort(out,CHEN_BOUNDED_ABORT_POSITION);return; }
    BoundedScratch &work=scratch[job_index];
    uint32_t sp=0;work.rules_error=0;
    work.positions[0]=job.root;
    work.nodes[0]=Node{NONE,0,NONE,0,job.root_ply,0,0,ChenRuleMove{},0};
    work.frames[0]=BoundedFrame{int32_t(job.depth),job.alpha,job.beta,-BOUNDED_INF,0,0,0,0};
    while(true) {
        BoundedFrame &frame=work.frames[sp];
        const ChenRulePosition &position=work.positions[sp];
        bool complete=false;
        if(!frame.entered) {
            if(out.nodes>=job.node_budget) { bounded_abort(out,CHEN_BOUNDED_ABORT_NODES);return; }
            ++out.nodes;
            const bool qsearch=job.quiescence && frame.depth<=0;
            if(qsearch) ++out.qnodes;
            frame.entered=1;work.pv_length[sp]=0;
            frame.count=legal_moves(position,work.moves[sp],&work.rules_error);
            if(work.rules_error) { bounded_abort(out,CHEN_BOUNDED_ABORT_RULES);return; }
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
                    bounded_abort(out,CHEN_BOUNDED_ABORT_QPLY);return;
                }
            }
        }
        if(!complete && (frame.alpha>=frame.beta || frame.cursor==frame.count)) complete=true;
        if(complete) {
            const int32_t value=frame.best;
            if(!sp) {
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
                if(length+1>CHEN_BOUNDED_STACK) { bounded_abort(out,CHEN_BOUNDED_ABORT_STACK);return; }
                work.pv[sp][0]=work.moves[sp][parent.cursor-1];
                for(uint32_t i=0;i<length;++i) work.pv[sp][i+1]=work.pv[child][i];
                work.pv_length[sp]=length+1;
            }
            if(score>parent.alpha) parent.alpha=score;
            continue;
        }
        if(sp+1>=CHEN_BOUNDED_STACK) { bounded_abort(out,CHEN_BOUNDED_ABORT_STACK);return; }
        const ChenRuleMove move=work.moves[sp][frame.cursor++];
        const bool qsearch=job.quiescence && frame.depth<=0;
        work.positions[sp+1]=apply<true>(position,move);
        work.nodes[sp+1]=Node{sp,0,NONE,0,job.root_ply+sp+1,0,0,move,0};
        work.frames[sp+1]=BoundedFrame{qsearch?0:frame.depth-1,-frame.beta,-frame.alpha,
            -BOUNDED_INF,qsearch?frame.qply+1:0,0,0,0};
        ++sp;
    }
}
int32_t bounded_failure(char *error,uint64_t capacity,const char *message) {
    if(error && capacity) std::snprintf(error,size_t(capacity),"%s",message);
    return -1;
}
}

extern "C" int32_t chen_bounded_search_run(int32_t device,const ChenBoundedJob *jobs,uint32_t count,
    const uint64_t *history,uint64_t history_count,ChenBoundedResult *results,
    char *error,uint64_t error_capacity) {
    if(error && error_capacity) error[0]=0;
    if(!jobs || !results || !count || count>CHEN_BOUNDED_MAX_JOBS || history_count>65535 ||
        (history_count && !history)) return bounded_failure(error,error_capacity,"invalid bounded batch buffers/counts");
    for(uint32_t i=0;i<count;++i) {
        const ChenBoundedJob &job=jobs[i];
        if(job.abi_version!=CHEN_BOUNDED_ABI || job.struct_size!=sizeof(job) || job.reserved ||
            job.depth>CHEN_BOUNDED_MAX_DEPTH || job.max_qply>CHEN_BOUNDED_MAX_QPLY ||
            job.root_ply>=CHEN_BOUNDED_STACK || job.depth+job.max_qply+job.root_ply>=CHEN_BOUNDED_STACK ||
            job.evaluator>1 || job.quiescence>1 || job.node_budget>CHEN_BOUNDED_MAX_NODES ||
            job.alpha<-BOUNDED_INF || job.alpha>=job.beta || job.beta>BOUNDED_INF ||
            job.history_offset>history_count || job.history_count>history_count-job.history_offset)
            return bounded_failure(error,error_capacity,"invalid bounded job configuration");
    }
    if(std::getenv("HSA_OVERRIDE_GFX_VERSION"))
        return bounded_failure(error,error_capacity,"architecture overrides forbidden");
    hipError_t code=hipSetDevice(device);
    if(code!=hipSuccess) return bounded_failure(error,error_capacity,hipGetErrorString(code));
    hipDeviceProp_t properties{};code=hipGetDeviceProperties(&properties,device);
    if(code!=hipSuccess) return bounded_failure(error,error_capacity,hipGetErrorString(code));
    if(std::strncmp(properties.gcnArchName,"gfx1030",7) ||
        (properties.gcnArchName[7] && properties.gcnArchName[7]!=':'))
        return bounded_failure(error,error_capacity,"bounded reference requires actual gfx1030");
    hipStream_t stream=nullptr;
    ChenBoundedJob *device_jobs=nullptr;ChenBoundedResult *device_results=nullptr;
    BoundedScratch *scratch=nullptr;uint64_t *device_history=nullptr;
    ChenBoundedResult staged[CHEN_BOUNDED_MAX_JOBS]{};
    code=hipStreamCreateWithFlags(&stream,hipStreamNonBlocking);
    if(code!=hipSuccess) return bounded_failure(error,error_capacity,hipGetErrorString(code));
    if(code==hipSuccess) code=hipMalloc(&device_jobs,size_t(count)*sizeof(*device_jobs));
    if(code==hipSuccess) code=hipMalloc(&device_results,size_t(count)*sizeof(*device_results));
    if(code==hipSuccess) code=hipMalloc(&scratch,size_t(count)*sizeof(*scratch));
    if(code==hipSuccess && history_count) code=hipMalloc(&device_history,size_t(history_count)*sizeof(uint64_t));
    if(code==hipSuccess) code=hipMemcpyAsync(device_jobs,jobs,size_t(count)*sizeof(*jobs),hipMemcpyHostToDevice,stream);
    if(code==hipSuccess && history_count)
        code=hipMemcpyAsync(device_history,history,size_t(history_count)*sizeof(uint64_t),hipMemcpyHostToDevice,stream);
    if(code==hipSuccess) {
        hipLaunchKernelGGL(bounded_search_kernel,dim3((count+63)/64),dim3(64),0,stream,
            device_jobs,count,device_history,scratch,device_results);
        code=hipGetLastError();
    }
    if(code==hipSuccess)
        code=hipMemcpyAsync(staged,device_results,size_t(count)*sizeof(*results),hipMemcpyDeviceToHost,stream);
    const hipError_t retired=hipStreamSynchronize(stream);
    if(retired!=hipSuccess) {
        // Do not free buffers potentially referenced by an unretired kernel.
        // This experimental API requires caller process retirement on HIP faults.
        return bounded_failure(error,error_capacity,"HIP retirement failed; retire caller process");
    }
    hipError_t cleanup=hipSuccess;
    void *buffers[]={device_jobs,device_results,scratch,device_history};
    for(void *buffer:buffers) if(buffer) {
        const hipError_t freed=hipFree(buffer);if(cleanup==hipSuccess) cleanup=freed;
    }
    const hipError_t destroyed=hipStreamDestroy(stream);if(cleanup==hipSuccess) cleanup=destroyed;
    if(code!=hipSuccess) return bounded_failure(error,error_capacity,hipGetErrorString(code));
    if(cleanup!=hipSuccess) return bounded_failure(error,error_capacity,hipGetErrorString(cleanup));
    // Publication is one host copy only after complete device retirement and
    // cleanup. A configuration/runtime failure leaves caller output untouched.
    std::memcpy(results,staged,size_t(count)*sizeof(*results));
    return 0;
}
