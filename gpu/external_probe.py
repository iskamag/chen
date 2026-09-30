#!/usr/bin/env python3
"""Actual device-only synthetic callback ABI boundary tests, not neural inference."""
import argparse
import ctypes as c
import json
from pathlib import Path
import chess
import torch
from rules_probe import Stats, Result, Move, bind, pack


class Owner(c.Structure):
    _fields_ = [("calls", c.c_uint64), ("mode", c.c_uint32), ("fail_at", c.c_uint32),
                ("score", c.c_int32), ("reserved", c.c_uint32)]


class Descriptor(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32), ("owner", c.c_void_p),
                ("evaluate", c.c_void_p), ("runtime_identity", c.c_uint64 * 5),
                ("max_batch", c.c_uint64), ("reserved", c.c_uint64)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("callback_library")
    args = parser.parse_args()
    assert torch.cuda.is_available() and c.sizeof(Descriptor) == 80
    runtime = c.CDLL(str(Path(torch.__file__).parent/'lib/libamdhip64.so'),mode=c.RTLD_GLOBAL)
    lib = bind(args.library)
    lib.chen_external_abi_version.restype=c.c_uint32
    assert lib.chen_external_abi_version()==1
    lib.chen_replay_runtime_identity.argtypes=[c.c_uint32]
    lib.chen_replay_runtime_identity.restype=c.c_uint64
    tokens=[lib.chen_replay_runtime_identity(i) for i in range(5)]
    for i,name in enumerate(('hipPointerGetAttributes','hipMemGetAddressRange','hipMalloc','hipFree','hipStreamSynchronize')):
        assert tokens[i]==c.cast(getattr(runtime,name),c.c_void_p).value
    callback=c.CDLL(args.callback_library)
    lib.chen_rules_evaluate_external.argtypes=[c.c_void_p,c.c_uint64,c.c_uint64,c.POINTER(Descriptor),c.POINTER(Stats),c.c_char_p,c.c_uint64]
    owner=Owner(0,3,2,17,0)
    desc=Descriptor(1,80,c.addressof(owner),c.cast(callback.chen_external_probe_callback,c.c_void_p).value,(c.c_uint64*5)(*tokens),7,0)
    error=c.create_string_buffer(1024)
    ctx=lib.chen_rules_create(0,128,error,len(error))
    assert ctx,error.value.decode()
    def call(name,*values,expected=0):
        status=getattr(lib,'chen_rules_'+name)(ctx,*values,error,len(error))
        assert status==expected,(name,status,error.value.decode())
    def evaluate(span,descriptor=desc,expected=0):
        stats=Stats();call('evaluate_external',*span,descriptor,stats,expected=expected);return stats
    try:
        root=pack(chess.Board());stats=Stats()
        call('reset',root,None,0);call('expand',0,1,1,stats)
        span=(stats.next_start,stats.next_count)
        # Second chunk fails after device writes; no static node scores commit.
        evaluate(span,expected=-1)
        call('classify',*span,1,stats)
        assert stats.evaluated_positions==0 and stats.classified_positions==21
        call('reduce',0,1,stats,expected=-1)
        owner.mode=0;owner.calls=0
        success=evaluate(span)
        assert owner.calls==3 and success.evaluated_positions==20
        repeated=evaluate(span)
        assert owner.calls==3 and repeated.evaluated_positions==20
        call('reduce',0,1,stats)
        result=Result();pv=(Move*8)()
        call('finish',result,pv,8,stats)
        assert result.score==-17 and result.pv_length==1
        # Descriptor identity and classical selector cannot change within reset.
        altered=Descriptor.from_buffer_copy(desc);altered.max_batch=8
        evaluate(span,altered,expected=-1)
        altered=Descriptor.from_buffer_copy(desc);altered.owner+=8
        evaluate(span,altered,expected=-1)
        altered=Descriptor.from_buffer_copy(desc);altered.runtime_identity[0]^=1
        evaluate(span,altered,expected=-1)
        call('evaluate',*span,0,stats,expected=-1)
        evaluate((0,1),expected=-1)
        for mode in (1,2):
            call('reset',root,None,0);call('expand',0,1,1,stats)
            owner.mode=mode;owner.calls=0
            evaluate(span,expected=-1)
            call('classify',*span,1,stats)
            assert stats.evaluated_positions==0
            owner.mode=0
            assert evaluate(span).evaluated_positions==20
        # Exact mate beats halfmove draw and does not invoke callback.
        mate=pack(chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 100 1'))
        call('reset',mate,None,0);owner.calls=0
        terminal=evaluate((0,1))
        assert owner.calls==0 and terminal.evaluated_positions==0 and terminal.classified_positions==1
        call('finish',result,pv,8,stats)
        assert result.score==-30000
        assert evaluate((1,0)).evaluated_positions==0 and owner.calls==0
        print(json.dumps({'record':'external_callback_contract','target':'gfx1030','device_only_synthetic_fixture':True,
                          'chunk_failure_no_static_commit':True,'missing_or_out_of_range_rejected':True,
                          'retry_success':True,'idempotent_no_callback':True,'terminal_and_empty_no_callback':True,
                          'mate_before_halfmove_draw':True,'descriptor_runtime_and_mode_locks':True,
                          'expanded_phase_rejected':True,'static_score17_root_negamax_minus17':True,
                          'scope':'GPU callback lifecycle/commit protocol only, not learned inference or strength.'}))
    finally:
        lib.chen_rules_destroy(ctx)


if __name__=='__main__':main()
