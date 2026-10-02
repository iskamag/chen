// Mechanism-only actual-device selector traces, not a legal chess strength test.
#include "chen_rules.hip"
#include <vector>

namespace {
__global__ void cache_trace(const Node *nodes,const uint32_t *counts,const SelectMeta *meta,
    uint32_t *reference_flags,uint32_t *cached_flags,SelectWork *reference,SelectWork *cached,
    uint64_t used,uint64_t extra,uint64_t remaining,uint32_t batch,uint32_t candidates,
    uint32_t depth,uint32_t *error) {
    selective_choose_device<false>(nodes,counts,meta,reference_flags,reference,used,used+extra,
        remaining,batch,candidates,depth,error);
    selective_choose_device<true>(nodes,counts,meta,cached_flags,cached,used,used+extra,
        remaining,batch,candidates,depth,error+1);
    // Increment preserves every upper field, including the sparse marker.
    uint32_t packed=SELECT_CACHE_VALID|SELECT_CACHE_OFFSET_MASK|SELECT_SPARSE_MARK;
    for(uint32_t i=0;i<CHEN_SELECTIVE_MAX_CANDIDATES;++i) selective_reserve<true>(packed);
    if(selective_reservations<true>(packed)!=CHEN_SELECTIVE_MAX_CANDIDATES ||
        (packed&~SELECT_RESERVATION_MASK)!=(SELECT_CACHE_VALID|SELECT_CACHE_OFFSET_MASK|SELECT_SPARSE_MARK))
        device_error(error+1,4);
}
void check(hipError_t status) {
    if(status!=hipSuccess) { std::fprintf(stderr,"cache fixture HIP error %s\n",hipGetErrorString(status));std::exit(1); }
}
void fixture(uint32_t depth,uint32_t batch,uint32_t candidates,uint64_t extra,uint64_t remaining,bool reject_all) {
    // A shallow 256-child node checks the complete offset field. A deep tree
    // has a shared unary chain then 8 groups of 256 leaves, exercising cache
    // reuse, exhaustion, 1024 rejected candidates and prefix trace comparison.
    const uint32_t chain=depth==2?0:depth-2,groups=depth==2?1:8;
    const uint64_t group_start=chain,leaf_start=chain+groups,used=leaf_start+uint64_t(groups)*256;
    std::vector<Node> nodes(used),after(used);std::vector<SelectMeta> meta(used);
    std::vector<uint32_t> counts(used),ref_flags(used),cached_flags(used);
    for(uint32_t ply=0;ply<chain;++ply) {
        const bool last=ply+1==chain;
        nodes[ply]=Node{ply?ply-1:NONE,last?group_start:ply+1,NONE,last?groups:1,ply,
            CLASSIFIED|SCORED|EXPANDED,0,ChenRuleMove{},0};
        meta[ply]=SelectMeta{uint64_t(groups)*256,depth-1,depth-1};
    }
    for(uint32_t g=0;g<groups;++g) {
        const uint64_t index=group_start+g;
        nodes[index]=Node{chain?chain-1:NONE,leaf_start+uint64_t(g)*256,NONE,256,depth-2,
            CLASSIFIED|SCORED|EXPANDED,0,ChenRuleMove{},0};
        meta[index]=SelectMeta{256,depth-1,depth-1};
        for(uint32_t j=0;j<256;++j) {
            const uint64_t child=nodes[index].first_child+j;
            nodes[child]=Node{index,0,NONE,0,depth-1,CLASSIFIED|SCORED,
                j==255?-1000:-int(j%7),ChenRuleMove{},0};
            meta[child]=SelectMeta{1,depth-1,depth-1};counts[child]=reject_all?2:1+(j%23);
        }
    }
    Node *device_nodes=nullptr;SelectMeta *device_meta=nullptr;uint32_t *device_counts=nullptr;
    uint32_t *a=nullptr,*b=nullptr,*error=nullptr;SelectWork *aw=nullptr,*bw=nullptr;
    check(hipMalloc(&device_nodes,used*sizeof(Node)));check(hipMalloc(&device_meta,used*sizeof(SelectMeta)));
    check(hipMalloc(&device_counts,used*4));check(hipMalloc(&a,used*4));check(hipMalloc(&b,used*4));
    check(hipMalloc(&error,8));check(hipMalloc(&aw,sizeof(SelectWork)));check(hipMalloc(&bw,sizeof(SelectWork)));
    check(hipMemcpy(device_nodes,nodes.data(),used*sizeof(Node),hipMemcpyHostToDevice));
    check(hipMemcpy(device_meta,meta.data(),used*sizeof(SelectMeta),hipMemcpyHostToDevice));
    check(hipMemcpy(device_counts,counts.data(),used*4,hipMemcpyHostToDevice));
    check(hipMemset(a,0,used*4));check(hipMemset(b,0,used*4));check(hipMemset(error,0,8));
    hipLaunchKernelGGL(cache_trace,dim3(1),dim3(1),0,0,device_nodes,device_counts,device_meta,a,b,aw,bw,
        used,extra,remaining,batch,candidates,depth,error);
    check(hipGetLastError());check(hipDeviceSynchronize());
    uint32_t errors[2];SelectWork ref{},cached{};
    check(hipMemcpy(errors,error,8,hipMemcpyDeviceToHost));
    check(hipMemcpy(&ref,aw,sizeof(ref),hipMemcpyDeviceToHost));check(hipMemcpy(&cached,bw,sizeof(cached),hipMemcpyDeviceToHost));
    check(hipMemcpy(ref_flags.data(),a,used*4,hipMemcpyDeviceToHost));check(hipMemcpy(cached_flags.data(),b,used*4,hipMemcpyDeviceToHost));
    check(hipMemcpy(after.data(),device_nodes,used*sizeof(Node),hipMemcpyDeviceToHost));
    if(errors[0] || errors[1] || std::memcmp(&ref.head,&cached.head,sizeof(SelectHead))) std::exit(2);
    if(std::memcmp(nodes.data(),after.data(),used*sizeof(Node))) std::exit(3);
    for(uint64_t i=0;i<used;++i) if(ref_flags[i]!=(cached_flags[i]&SELECT_RESERVATION_MASK)) std::exit(4);
    for(uint32_t i=0;i<ref.head.selected;++i)
        if(ref.selected[i]!=cached.selected[i] || ref.offsets[i]!=cached.offsets[i]) std::exit(5);
    std::printf("{\"depth\":%u,\"batch\":%u,\"candidate_limit\":%u,\"extra_capacity\":%llu,\"remaining\":%llu,\"reject_all\":%s,\"selected\":%u,\"examined\":%u,\"added\":%u,\"stop\":%u,\"status\":\"passed\",\"selected_indices\":[",
        depth,batch,candidates,static_cast<unsigned long long>(extra),static_cast<unsigned long long>(remaining),
        reject_all?"true":"false",ref.head.selected,ref.head.examined,ref.head.added,ref.head.stop);
    for(uint32_t i=0;i<ref.head.selected;++i) std::printf("%s%llu",i?",":"",static_cast<unsigned long long>(ref.selected[i]));
    std::printf("],\"offsets\":[");
    for(uint32_t i=0;i<ref.head.selected;++i) std::printf("%s%llu",i?",":"",static_cast<unsigned long long>(ref.offsets[i]));
    std::printf("]}\n");
    check(hipFree(device_nodes));check(hipFree(device_meta));check(hipFree(device_counts));
    check(hipFree(a));check(hipFree(b));check(hipFree(error));check(hipFree(aw));check(hipFree(bw));
}
}
int main() {
    check(hipSetDevice(0));
    for(uint32_t depth:{2u,CHEN_SELECTIVE_MAX_PLY}) for(uint32_t batch:{1u,32u,256u})
        for(uint32_t candidates:{batch,CHEN_SELECTIVE_MAX_CANDIDATES}) {
            fixture(depth,batch,candidates,65536,65536,false);
            fixture(depth,batch,candidates,4096,17,false);
            fixture(depth,batch,candidates,17,4096,false);
            fixture(depth,batch,candidates,4096,1,true);
        }
    return 0;
}
