// Mechanism-only backup fixture: synthetic trees, not legal chess positions.
// Include the exact candidate TU so production private types/functions and
// scratch sizes are tested rather than reimplemented in this fixture.
#include "chen_rules.hip"
#include <vector>

namespace {
struct GuardWork { uint64_t before[4];SelectWork work;uint64_t after[4]; };
constexpr uint64_t GUARD=0x6368656e73706172ull;
__global__ void sparse_fixture(Node *reference,SelectMeta *ref_meta,Node *candidate,
    SelectMeta *candidate_meta,uint32_t *reserved,GuardWork *guard,uint64_t old_used,
    uint64_t used,uint32_t depth,uint32_t *ref_error,uint32_t *candidate_error) {
    selective_backup_device(reference,ref_meta,reserved,old_used,used,ref_error);
    selective_backup_sparse_device(candidate,candidate_meta,reserved,&guard->work,
        old_used,used,depth,candidate_error);
}
void check(hipError_t status) {
    if(status!=hipSuccess) { std::fprintf(stderr,"fixture HIP error: %s\n",hipGetErrorString(status));std::exit(1); }
}
void fixture(uint32_t depth,uint32_t batch) {
    const uint64_t old_used=1+uint64_t(depth-1)*batch,used=old_used+batch;
    std::vector<Node> input(used),reference(used),candidate(used);
    std::vector<SelectMeta> meta(used),ref_meta(used),candidate_meta(used);
    std::vector<uint32_t> reserved(used);
    GuardWork host{};
    for(uint32_t i=0;i<4;++i) host.before[i]=host.after[i]=GUARD;
    host.work.head=SelectHead{batch,batch,batch,CHEN_SELECTIVE_PROGRESS};
    // Root, then breadth-first fixed-width unary chains. Each root child is
    // one independent chain; admitted parents are the last old ply.
    input[0]=Node{NONE,depth==1?old_used:1,NONE,batch,0,CLASSIFIED|SCORED|EXPANDED,0,ChenRuleMove{},0};
    meta[0]=SelectMeta{batch,depth-1,depth-1};reserved[0]=batch;
    for(uint32_t ply=1;ply<depth;++ply) for(uint32_t lane=0;lane<batch;++lane) {
        const uint64_t index=1+uint64_t(ply-1)*batch+lane;
        const uint64_t parent=ply==1?0:index-batch;
        const uint64_t child=ply+1==depth?old_used+lane:index+batch;
        input[index]=Node{parent,child,NONE,1,ply,CLASSIFIED|SCORED|EXPANDED,0,ChenRuleMove{},0};
        meta[index]=SelectMeta{1,depth-1,depth-1};reserved[index]=1;
    }
    for(uint32_t lane=0;lane<batch;++lane) {
        const uint64_t parent=depth==1?0:1+uint64_t(depth-2)*batch+lane;
        input[old_used+lane]=Node{parent,0,NONE,0,depth,CLASSIFIED|SCORED,int(lane%29)-14,ChenRuleMove{},0};
        meta[old_used+lane]=SelectMeta{0,depth,depth};
        host.work.selected[lane]=parent;host.work.offsets[lane]=lane;
    }
    Node *a=nullptr,*b=nullptr;SelectMeta *am=nullptr,*bm=nullptr;
    uint32_t *flags=nullptr,*ae=nullptr,*be=nullptr;GuardWork *work=nullptr;
    check(hipMalloc(&a,used*sizeof(Node)));check(hipMalloc(&b,used*sizeof(Node)));
    check(hipMalloc(&am,used*sizeof(SelectMeta)));check(hipMalloc(&bm,used*sizeof(SelectMeta)));
    check(hipMalloc(&flags,used*sizeof(uint32_t)));check(hipMalloc(&work,sizeof(GuardWork)));
    check(hipMalloc(&ae,sizeof(uint32_t)));check(hipMalloc(&be,sizeof(uint32_t)));
    check(hipMemcpy(a,input.data(),used*sizeof(Node),hipMemcpyHostToDevice));
    check(hipMemcpy(b,input.data(),used*sizeof(Node),hipMemcpyHostToDevice));
    check(hipMemcpy(am,meta.data(),used*sizeof(SelectMeta),hipMemcpyHostToDevice));
    check(hipMemcpy(bm,meta.data(),used*sizeof(SelectMeta),hipMemcpyHostToDevice));
    check(hipMemcpy(flags,reserved.data(),used*sizeof(uint32_t),hipMemcpyHostToDevice));
    check(hipMemcpy(work,&host,sizeof(host),hipMemcpyHostToDevice));
    check(hipMemset(ae,0,sizeof(uint32_t)));check(hipMemset(be,0,sizeof(uint32_t)));
    hipLaunchKernelGGL(sparse_fixture,dim3(1),dim3(1),0,0,a,am,b,bm,flags,work,old_used,used,depth,ae,be);
    check(hipGetLastError());check(hipDeviceSynchronize());
    uint32_t errors[2];check(hipMemcpy(&errors[0],ae,4,hipMemcpyDeviceToHost));check(hipMemcpy(&errors[1],be,4,hipMemcpyDeviceToHost));
    check(hipMemcpy(reference.data(),a,used*sizeof(Node),hipMemcpyDeviceToHost));
    check(hipMemcpy(candidate.data(),b,used*sizeof(Node),hipMemcpyDeviceToHost));
    check(hipMemcpy(ref_meta.data(),am,used*sizeof(SelectMeta),hipMemcpyDeviceToHost));
    check(hipMemcpy(candidate_meta.data(),bm,used*sizeof(SelectMeta),hipMemcpyDeviceToHost));
    check(hipMemcpy(&host,work,sizeof(host),hipMemcpyDeviceToHost));
    if(errors[0] || errors[1]) { std::fprintf(stderr,"fixture device errors %u/%u\n",errors[0],errors[1]);std::exit(2); }
    for(uint32_t i=0;i<4;++i) if(host.before[i]!=GUARD || host.after[i]!=GUARD) std::exit(3);
    for(uint64_t index=0;index<used;++index) {
        const Node &x=reference[index],&y=candidate[index];
        const SelectMeta &xm=ref_meta[index],&ym=candidate_meta[index];
        if(x.score!=y.score || x.best!=y.best || x.flags!=y.flags ||
            xm.open!=ym.open || xm.minimum!=ym.minimum || xm.maximum!=ym.maximum) std::exit(4);
    }
    uint64_t touched=0;
    for(uint32_t ply=0;ply<depth;++ply) {
        const uint32_t expected=ply?batch:1;
        if(host.work.sparse.counts[ply]!=expected) std::exit(5);
        touched+=host.work.sparse.counts[ply];
    }
    if(touched!=old_used) std::exit(6);
    std::printf("{\"depth\":%u,\"batch\":%u,\"old_nodes\":%llu,\"compared_nodes\":%llu,\"status\":\"passed\",\"guard_words\":8}\n",
        depth,batch,static_cast<unsigned long long>(old_used),static_cast<unsigned long long>(used));
    check(hipFree(a));check(hipFree(b));check(hipFree(am));check(hipFree(bm));
    check(hipFree(flags));check(hipFree(work));check(hipFree(ae));check(hipFree(be));
}
}
int main() {
    check(hipSetDevice(0));
    for(uint32_t depth:{1u,2u,CHEN_SELECTIVE_MAX_PLY})
        for(uint32_t batch:{1u,32u,CHEN_SELECTIVE_MAX_BATCH}) {
            // At ply1 only root may be selected; a selected-parent list cannot
            // contain root multiple times under the real scheduler contract.
            if(depth==1 && batch!=1) continue;
            fixture(depth,batch);
        }
    return 0;
}
