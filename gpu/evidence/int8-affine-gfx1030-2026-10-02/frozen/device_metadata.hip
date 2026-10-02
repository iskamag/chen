#include "chen-int8-linear.hip"
#include <cstdio>
int main() {
 hipDeviceProp_t p{}; hipError_t e=hipGetDeviceProperties(&p,0); if(e)return int(e);
 std::printf("{\"name\":\"%s\",\"gcnArchName\":\"%s\",\"warpSize\":%d,\"multiProcessorCount\":%d,\"maxThreadsPerMultiProcessor\":%d,\"maxBlocksPerMultiProcessor\":%d,\"regsPerMultiprocessor\":%d,\"cooperativeLaunch\":%d,\"cooperativeMultiDeviceLaunch\":%d,\"occupancy\":[",p.name,p.gcnArchName,p.warpSize,p.multiProcessorCount,p.maxThreadsPerMultiProcessor,p.maxBlocksPerMultiProcessor,p.regsPerMultiprocessor,p.cooperativeLaunch,p.cooperativeMultiDeviceLaunch);
 bool first=true;
 for(int block:{64,128,256}) {
  int packed=0,dynamic=0,scalar_=0;
  e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&packed,linear<true>,block,0); if(e)return int(e);
  e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&dynamic,dynamic_linear<false>,block,0); if(e)return int(e);
  e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&scalar_,scalar<true>,block,0); if(e)return int(e);
  std::printf("%s{\"block\":%d,\"packed_blocks_per_reported_multiprocessor\":%d,\"dynamic_blocks_per_reported_multiprocessor\":%d,\"scalar_blocks_per_reported_multiprocessor\":%d}",first?"":",",block,packed,dynamic,scalar_);first=false;
 }
 std::puts("]}");
}
