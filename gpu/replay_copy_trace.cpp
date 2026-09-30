// Diagnostic LD_PRELOAD interposition for the synchronous export copy boundary.
// Tracks hipMemcpy/hipMemcpyAsync while enabled; not a replacement for a full
// profiler. Probe state inspections occur outside the marked interval.
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <cstdint>
#include <dlfcn.h>
namespace {
std::atomic<bool> enabled{false};
std::atomic<uint64_t> calls[5]{},bytes[5]{},largest{0};
void record(size_t count,hipMemcpyKind kind) {
    if(!enabled.load()) return;
    const int index=int(kind);
    if(index<0 || index>=5) return;
    ++calls[index];bytes[index]+=count;
    uint64_t current=largest.load();
    while(current<count && !largest.compare_exchange_weak(current,count)) {}
}
}
struct ChenCopyTrace { uint64_t calls[5],bytes[5],largest; };
extern "C" void chen_copy_trace_begin() {
    enabled=false;
    for(int i=0;i<5;++i) {calls[i]=0;bytes[i]=0;}
    largest=0;enabled=true;
}
extern "C" void chen_copy_trace_end(ChenCopyTrace *out) {
    enabled=false;
    for(int i=0;i<5;++i) {out->calls[i]=calls[i].load();out->bytes[i]=bytes[i].load();}
    out->largest=largest.load();
}
extern "C" hipError_t hipMemcpy(void *dst,const void *src,size_t size,hipMemcpyKind kind) {
    using Function=hipError_t (*)(void *,const void *,size_t,hipMemcpyKind);
    static auto real=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"hipMemcpy"));
    if(!real) return hipErrorSharedObjectSymbolNotFound;
    record(size,kind);return real(dst,src,size,kind);
}
extern "C" hipError_t hipMemcpyAsync(void *dst,const void *src,size_t size,
    hipMemcpyKind kind,hipStream_t stream) {
    using Function=hipError_t (*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t);
    static auto real=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"hipMemcpyAsync"));
    if(!real) return hipErrorSharedObjectSymbolNotFound;
    record(size,kind);return real(dst,src,size,kind,stream);
}
