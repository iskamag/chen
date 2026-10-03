// Research-only append to a frozen chen_rules.hip. Original production kernels
// remain present; staging appends exact device-callable helper copies first.
#include "device_epoch.h"
#include <hip/hip_cooperative_groups.h>
#include <chrono>
namespace {
struct EpochRequest { uint64_t limit;uint32_t evaluator,advances; };
struct EpochState {
    EpochRequest request{};
    uint64_t used=1,steps=0,expanded=0;
    uint32_t active=0,attempts=0,error=0,reserved=0;
    SelectCompletion pending{};
    ChenEpochTrace last{},trace[4]{};
};
struct EpochData {
    ChenRulePosition *positions;Node *nodes;ChenRuleMove *moves;
    uint32_t *counts,*reservations;SelectMeta *meta;SelectWork *work;
    unsigned long long *classified,*evaluated;uint32_t *error;
    const uint64_t *history;uint64_t history_count,capacity;
    ChenSelectiveConfig config;
};
__device__ void epoch_begin_device(EpochState *state) {
    state->attempts=0;
    state->active=state->error?0u:1u;
    for(uint32_t i=0;i<4;++i) state->trace[i]=ChenEpochTrace{};
    if(state->request.advances!=2 && state->request.advances!=4) {
        state->error=4;state->active=0;
    }
}
__device__ void epoch_clear_device(const EpochData &d,EpochState *s,uint64_t row,uint64_t stride) {
    if(s->active) for(uint64_t index=row;index<s->used;index+=stride) d.reservations[index]=0;
}
__device__ void epoch_choose_device(const EpochData &d,EpochState *s) {
    if(!s->active) return;
    if(*d.error) { s->error=*d.error;s->active=0;return; }
    const uint32_t bootstrap=*d.classified==0;
    if(bootstrap && s->request.limit)
        epoch_classify_kernel(d.positions,d.nodes,d.moves,d.counts,0,1,1,
            d.history,d.history_count,d.classified,d.error,0);
    epoch_selective_choose_classical(d.nodes,d.counts,d.meta,d.classified,
        d.reservations,d.work,s->used,d.capacity,s->request.limit,bootstrap,
        d.config.batch_leaves,d.config.candidate_limit,d.config.max_ply,d.error);
}
__device__ void epoch_children_device(const EpochData &d,EpochState *s,uint64_t row,uint64_t stride) {
    if(!s->active) return;
    const uint64_t bound=uint64_t(d.config.batch_leaves)*CHEN_RULES_MAX_MOVES;
    for(uint64_t index=row;index<bound;index+=stride)
        epoch_selective_children_classical(d.positions,d.nodes,d.moves,d.counts,d.meta,d.work,
            s->used,d.capacity,s->request.limit,d.config.batch_leaves,d.config.candidate_limit,
            d.config.max_ply,s->request.evaluator,d.history,d.history_count,
            d.classified,d.evaluated,d.error,index);
}
__device__ void epoch_complete_device(const EpochData &d,EpochState *s) {
    if(!s->active) return;
    epoch_selective_complete_classical(d.nodes,d.meta,d.reservations,d.work,d.classified,d.evaluated,
        &s->pending,s->used,s->steps,s->expanded,d.config.max_ply,d.error);
    if(s->pending.error) { s->error=s->pending.error;s->active=0;return; }
    const ChenSelectiveStats stats=s->pending.stats;
    if(s->pending.reserved || stats.abi_version!=1 || stats.struct_size!=sizeof(stats) || stats.reserved ||
        stats.last_selected>d.config.batch_leaves || stats.last_added<stats.last_selected ||
        stats.arena_positions!=s->used+stats.last_added || stats.arena_positions>d.capacity ||
        stats.classified_positions>s->request.limit || stats.classified_positions<stats.last_added ||
        stats.evaluated_positions>stats.classified_positions ||
        (stats.classified_positions!=stats.arena_positions && !(stats.classified_positions==0 && stats.arena_positions==1)) ||
        stats.completed_steps!=s->steps+(stats.last_selected?1:0) ||
        stats.expanded_leaves!=s->expanded+stats.last_selected || stats.stop_reason>CHEN_SELECTIVE_FALLBACK ||
        s->attempts>=s->request.advances || s->attempts>=4) {
        device_error(d.error,4);s->error=4;s->active=0;return;
    }
    ChenEpochTrace candidate{};candidate.stats=stats;
    if(stats.root_flags&CHEN_SELECTIVE_ROOT_PUBLISHED) {
        candidate.result.score=d.nodes[0].score;
        uint64_t index=0;
        while(d.nodes[index].best!=NONE) {
            index=d.nodes[index].best;
            if(index>=stats.arena_positions || candidate.result.pv_length>=d.config.max_ply) {
                device_error(d.error,4);s->error=4;s->active=0;return;
            }
            candidate.pv[candidate.result.pv_length++]=d.nodes[index].move;
        }
        if(candidate.result.pv_length!=stats.pv_depth) {
            device_error(d.error,4);s->error=4;s->active=0;return;
        }
    }
    // Only fully scored/backed-up/validated data replaces the last publication.
    s->last=candidate;s->trace[s->attempts++]=candidate;
    s->used=stats.arena_positions;s->steps=stats.completed_steps;s->expanded=stats.expanded_leaves;
    s->active=(stats.stop_reason==CHEN_SELECTIVE_PROGRESS && s->attempts<s->request.advances)?1u:0u;
}
__global__ void epoch_begin(EpochData,EpochState *s) { epoch_begin_device(s); }
__global__ void epoch_clear(EpochData d,EpochState *s) {
    epoch_clear_device(d,s,uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,uint64_t(gridDim.x)*blockDim.x);
}
__global__ void epoch_choose(EpochData d,EpochState *s) { epoch_choose_device(d,s); }
__global__ void epoch_children(EpochData d,EpochState *s) {
    epoch_children_device(d,s,uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,uint64_t(gridDim.x)*blockDim.x);
}
__global__ void epoch_complete(EpochData d,EpochState *s) { epoch_complete_device(d,s); }
template<bool Cooperative> __global__ void epoch_resident(EpochData d,EpochState *s) {
    auto whole=cooperative_groups::this_grid();
    const uint64_t row=Cooperative?uint64_t(blockIdx.x)*blockDim.x+threadIdx.x:threadIdx.x;
    const uint64_t stride=Cooperative?uint64_t(gridDim.x)*blockDim.x:blockDim.x;
    if(!row) epoch_begin_device(s);
    if constexpr(Cooperative) whole.sync();else __syncthreads();
    // All threads execute every barrier, including inactive/failure paths.
    for(uint32_t phase=0;phase<s->request.advances;++phase) {
        epoch_clear_device(d,s,row,stride);
        if constexpr(Cooperative) whole.sync();else __syncthreads();
        if(!row) epoch_choose_device(d,s);
        if constexpr(Cooperative) whole.sync();else __syncthreads();
        epoch_children_device(d,s,row,stride);
        if constexpr(Cooperative) whole.sync();else __syncthreads();
        if(!row) epoch_complete_device(d,s);
        if constexpr(Cooperative) whole.sync();else __syncthreads();
    }
}
}
struct ChenDeviceEpochContext {
    ChenSelectiveContext *tree=nullptr;EpochState *state=nullptr,*host=nullptr;
    EpochData data{};hipGraph_t graph[2]={};hipGraphExec_t executable[2]={};
    uint32_t mode=0,block=64,resident_blocks=0,grid_blocks=1;double setup_ms=0;
};
namespace {
void epoch_destroy_graph(ChenDeviceEpochContext *c) {
    for(uint32_t i=0;i<2;++i) {
        if(c->executable[i]) selective_retire(hipGraphExecDestroy(c->executable[i]));
        if(c->graph[i]) selective_retire(hipGraphDestroy(c->graph[i]));
        c->executable[i]=nullptr;c->graph[i]=nullptr;
    }
}
hipError_t epoch_graph_node(ChenDeviceEpochContext *c,uint32_t graph_index,const void *function,
                          uint32_t grid,uint32_t block,hipGraphNode_t &previous,bool first=false) {
    void *arguments[]={&c->data,&c->state};hipKernelNodeParams params{};
    params.func=const_cast<void *>(function);params.gridDim=dim3(grid);params.blockDim=dim3(block);
    params.kernelParams=arguments;
    hipGraphNode_t node=nullptr;
    const hipError_t result=hipGraphAddKernelNode(&node,c->graph[graph_index],first?nullptr:&previous,first?0:1,&params);
    if(result==hipSuccess) previous=node;
    return result;
}
int32_t epoch_error(ChenDeviceEpochContext *c,char *error,uint64_t cap,const char *message) {
    if(c&&c->tree) c->tree->valid=false;
    return fail(error,cap,message);
}
}
extern "C" void chen_epoch_destroy(ChenDeviceEpochContext *c) {
    if(!c) return;
    if(c->tree) {
        selective_retire(hipSetDevice(c->tree->arena->device));
        selective_retire(hipStreamSynchronize(c->tree->arena->stream));
        epoch_destroy_graph(c);
        if(c->state) selective_retire(hipFree(c->state));
        if(c->host) selective_retire(hipHostFree(c->host));
        chen_selective_destroy(c->tree);
    }
    delete c;
}
extern "C" ChenDeviceEpochContext *chen_epoch_create(int32_t device,uint64_t capacity,
    uint32_t mode,uint32_t block,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(mode>3 || (block!=64 && block!=128 && block!=256)) {
        fail(error,cap,"invalid epoch mode/block size");return nullptr;
    }
    auto *c=new(std::nothrow) ChenDeviceEpochContext;
    if(!c) { fail(error,cap,"epoch context allocation failed");return nullptr; }
    c->mode=mode;c->block=block;c->tree=chen_selective_create(device,capacity,error,cap);
    if(!c->tree) { delete c;return nullptr; }
    hipError_t code=hipMalloc(&c->state,sizeof(EpochState));
    if(code==hipSuccess) code=hipHostMalloc(&c->host,sizeof(EpochState));
    if(code!=hipSuccess) { hip_fail(error,cap,code);chen_epoch_destroy(c);return nullptr; }
    return c;
}
extern "C" int32_t chen_epoch_reset(ChenDeviceEpochContext *c,const ChenRulePosition *root,
    const uint64_t *history,uint64_t history_count,const ChenSelectiveConfig *config,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!c) return fail(error,cap,"null epoch context");
    ChenSelectiveStats initial{};
    int32_t status=chen_selective_reset(c->tree,root,history,history_count,config,&initial,error,cap);
    if(status) return status;
    epoch_destroy_graph(c);
    auto *a=c->tree->arena;
    c->data=EpochData{a->positions,a->nodes,a->moves,a->counts,a->flags,c->tree->meta,c->tree->work,
        a->classified,a->evaluated,&a->summary->error,a->history,a->history_count,a->capacity,*config};
    *c->host=EpochState{};c->host->last.stats=initial;
    hipError_t code=hipMemcpy(c->state,c->host,sizeof(EpochState),hipMemcpyHostToDevice);
    if(code!=hipSuccess) return selective_status(c->tree,hip_fail(error,cap,code));
    const auto before=std::chrono::steady_clock::now();
    if(c->mode==0) {
        const uint64_t children=uint64_t(config->batch_leaves)*CHEN_RULES_MAX_MOVES;
        const uint32_t child_grid=uint32_t((children+c->block-1)/c->block);
        const uint32_t clear_grid=uint32_t(((a->capacity<65536?a->capacity:65536)+c->block-1)/c->block);
        for(uint32_t graph_index=0;graph_index<2 && code==hipSuccess;++graph_index) {
            code=hipGraphCreate(&c->graph[graph_index],0);hipGraphNode_t previous=nullptr;
            if(code==hipSuccess) code=epoch_graph_node(c,graph_index,reinterpret_cast<const void *>(epoch_begin),1,1,previous,true);
            for(uint32_t i=0;i<(graph_index?4u:2u) && code==hipSuccess;++i) {
                code=epoch_graph_node(c,graph_index,reinterpret_cast<const void *>(epoch_clear),clear_grid,c->block,previous);
                if(code==hipSuccess) code=epoch_graph_node(c,graph_index,reinterpret_cast<const void *>(epoch_choose),1,1,previous);
                if(code==hipSuccess) code=epoch_graph_node(c,graph_index,reinterpret_cast<const void *>(epoch_children),child_grid,c->block,previous);
                if(code==hipSuccess) code=epoch_graph_node(c,graph_index,reinterpret_cast<const void *>(epoch_complete),1,1,previous);
            }
            if(code==hipSuccess) code=hipGraphInstantiate(&c->executable[graph_index],c->graph[graph_index],nullptr,nullptr,0);
        }
    } else if(c->mode==2) {
        hipDeviceProp_t properties{};int resident=0;
        code=hipGetDeviceProperties(&properties,a->device);
        if(code==hipSuccess && !properties.cooperativeLaunch)
            return epoch_error(c,error,cap,"actual device does not support cooperative launch");
        if(code==hipSuccess) code=hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,epoch_resident<true>,c->block,0);
        if(code==hipSuccess && (resident<=0 || properties.multiProcessorCount<=0))
            return epoch_error(c,error,cap,"actual combined epoch kernel has no resident occupancy");
        c->resident_blocks=uint32_t(resident);
        const uint64_t required=(uint64_t(config->batch_leaves)*CHEN_RULES_MAX_MOVES+c->block-1)/c->block;
        const uint64_t resident_grid=uint64_t(resident)*properties.multiProcessorCount;
        c->grid_blocks=uint32_t(required<resident_grid?required:resident_grid);
    }
    c->setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    if(code!=hipSuccess) return selective_status(c->tree,hip_fail(error,cap,code));
    return 0;
}
extern "C" int32_t chen_epoch_advance(ChenDeviceEpochContext *c,uint64_t limit,uint32_t evaluator,
    uint32_t advances,ChenEpochReport *report,char *error,uint64_t cap) {
    clear_error(error,cap);
    if(!c || !c->tree->valid) return fail(error,cap,"epoch context must reset before advance");
    auto *t=c->tree;auto *a=t->arena;
    if(!report || evaluator>1 || (advances!=2 && advances!=4) || limit<t->classified)
        return epoch_error(c,error,cap,"invalid epoch admission");
    if(t->evaluator>=0 && t->evaluator!=int(evaluator)) return epoch_error(c,error,cap,"epoch evaluator changed before reset");
    int32_t status=lock_mode(a,1,error,cap);if(status) return selective_status(t,status);
    if(a->evaluator>=0 && a->evaluator!=int(evaluator)) return epoch_error(c,error,cap,"resident evaluator changed before reset");
    a->evaluator=t->evaluator=int(evaluator);
    status=prepare(a,error,cap);if(status) return selective_status(t,status);
    const EpochRequest request{limit,evaluator,advances};
    hipError_t code=hipMemcpyAsync(&c->state->request,&request,sizeof(request),hipMemcpyHostToDevice,a->stream);
    if(code==hipSuccess) {
        if(c->mode==0) code=hipGraphLaunch(c->executable[advances==4],a->stream);
        else if(c->mode==3) {
            // Causal control: exactly the graph's functions, dimensions, order,
            // request/copy/fence and publication; only host dispatch differs.
            const uint64_t children=uint64_t(c->data.config.batch_leaves)*CHEN_RULES_MAX_MOVES;
            const uint32_t child_grid=uint32_t((children+c->block-1)/c->block);
            const uint32_t clear_grid=uint32_t(((a->capacity<65536?a->capacity:65536)+c->block-1)/c->block);
            hipLaunchKernelGGL(epoch_begin,dim3(1),dim3(1),0,a->stream,c->data,c->state);
            code=hipGetLastError();
            for(uint32_t i=0;i<advances && code==hipSuccess;++i) {
                hipLaunchKernelGGL(epoch_clear,dim3(clear_grid),dim3(c->block),0,a->stream,c->data,c->state);
                code=hipGetLastError();
                if(code==hipSuccess) {
                    hipLaunchKernelGGL(epoch_choose,dim3(1),dim3(1),0,a->stream,c->data,c->state);
                    code=hipGetLastError();
                }
                if(code==hipSuccess) {
                    hipLaunchKernelGGL(epoch_children,dim3(child_grid),dim3(c->block),0,a->stream,c->data,c->state);
                    code=hipGetLastError();
                }
                if(code==hipSuccess) {
                    hipLaunchKernelGGL(epoch_complete,dim3(1),dim3(1),0,a->stream,c->data,c->state);
                    code=hipGetLastError();
                }
            }
        } else if(c->mode==1) {
            hipLaunchKernelGGL((epoch_resident<false>),dim3(1),dim3(c->block),0,a->stream,c->data,c->state);
            code=hipGetLastError();
        } else {
            void *arguments[]={&c->data,&c->state};
            code=hipLaunchCooperativeKernel(reinterpret_cast<const void *>(epoch_resident<true>),
                dim3(c->grid_blocks),dim3(c->block),arguments,0,a->stream);
        }
    }
    if(code==hipSuccess) code=hipEventRecord(a->end,a->stream);
    if(code==hipSuccess) code=hipMemcpyAsync(c->host,c->state,sizeof(EpochState),hipMemcpyDeviceToHost,a->stream);
    const hipError_t sync=hipStreamSynchronize(a->stream);if(code==hipSuccess) code=sync;
    float elapsed=0;if(code==hipSuccess) code=hipEventElapsedTime(&elapsed,a->begin,a->end);
    if(code!=hipSuccess) return selective_status(t,hip_fail(error,cap,code));
    if(c->host->error) return epoch_error(c,error,cap,"sticky device epoch failure");
    if(!c->host->attempts || c->host->attempts>advances || c->host->reserved)
        return epoch_error(c,error,cap,"invalid epoch completion packet");
    const auto &s=c->host->last.stats;
    if(s.arena_positions!=c->host->used || s.completed_steps!=c->host->steps ||
       s.expanded_leaves!=c->host->expanded || s.classified_positions< t->classified)
        return epoch_error(c,error,cap,"nonmonotone epoch completion counters");
    a->kernel_total_ms+=elapsed;a->used=s.arena_positions;
    t->root_classified=s.classified_positions!=0;t->classified=s.classified_positions;
    t->steps=s.completed_steps;t->expanded=s.expanded_leaves;
    t->selected=s.last_selected;t->added=s.last_added;t->stop=s.stop_reason;
    ChenEpochReport out{};out.abi_version=1;out.struct_size=sizeof(out);
    out.attempts=c->host->attempts;out.mode=c->mode;out.advance_bound=advances;
    out.resident_blocks_per_multiprocessor=c->resident_blocks;out.grid_blocks=c->grid_blocks;out.setup_ms=c->setup_ms;
    for(uint32_t i=0;i<out.attempts;++i) out.trace[i]=c->host->trace[i];
    *report=out;return 0;
}
