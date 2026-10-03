import ctypes as c, hashlib, json, pathlib, sys
p=pathlib.Path(sys.argv[1]); lib=c.CDLL(str(p/'libchen_bounded_search.so'))
class Info(c.Structure):
 _fields_=[('name',c.c_char*128),('architecture',c.c_char*64),('total_memory',c.c_uint64)]+[(n,c.c_uint32) for n in ('compute_units','wave_size','async_engines','runtime_version')]
assert c.sizeof(Info)==216
lib.chen_gpu_create.argtypes=[c.c_int32,c.c_uint64,c.c_char_p,c.c_uint64]; lib.chen_gpu_create.restype=c.c_void_p
lib.chen_gpu_device_info.argtypes=[c.c_void_p,c.POINTER(Info),c.c_char_p,c.c_uint64]; lib.chen_gpu_device_info.restype=c.c_int32
lib.chen_gpu_destroy.argtypes=[c.c_void_p]; lib.chen_gpu_destroy.restype=None
err=c.create_string_buffer(1024); ctx=lib.chen_gpu_create(0,1,err,1024); assert ctx,err.value
info=Info()
try:
 assert lib.chen_gpu_device_info(ctx,c.byref(info),err,1024)==0,err.value
 providers=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(s in line for s in ('libamdhip64','libhsa-runtime64','libdrm_amdgpu')) and line.split()[-1].startswith('/')})
 result={n:getattr(info,n) for n in ('total_memory','compute_units','wave_size','async_engines','runtime_version')}
 result.update(name=info.name.decode(),architecture=info.architecture.decode(),providers={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in providers})
 print(json.dumps(result,indent=2))
finally:
 lib.chen_gpu_destroy(ctx)
