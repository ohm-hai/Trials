"""Phase 8 fused vs prod: focused comparison at bs=4,16,64,128."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel_fused import (
    moe_decode_megakernel_stage1_fused, moe_decode_megakernel_stage2_fused)
from aiter.fused_moe import fused_moe, QuantType, ActivationType

def bench_fn(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=torch.cuda.Event(enable_timing=True); t1=torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1)*1e3/iters

def moe_fused(h,w13f,w13s,w2f,w2s,tw,ti,BM):
    M,K=h.shape
    st,ei,ntp=moe_align_block_size(ti,BM,E); npm=ei.shape[0]
    a1f,a1s=per_token_group_quant_fp8(h,128)
    inter=torch.zeros(npm*BM,N_INTER,dtype=torch.bfloat16,device=DEV)
    moe_decode_megakernel_stage1_fused(a1f,w13f,a1s,w13s,st,ei,inter,TOPK,block_m=BM,block_n=128,block_k=128,fused_silu=True)
    a2f,a2s=per_token_group_quant_fp8(inter,128)
    out=torch.zeros(M,K,dtype=torch.float32,device=DEV)
    moe_decode_megakernel_stage2_fused(a2f,w2f,a2s,w2s,st,ei,out,tw.to(torch.float32).reshape(-1),TOPK,block_m=BM,block_n=128,block_k=128)
    return out.to(torch.bfloat16)

def main():
    torch.manual_seed(0)
    print("=== Phase 8 fused vs prod (focused) ===")
    print(f"{'bs':>5} {'BM':>4} {'fused us':>9} {'prod us':>9} {'spd':>7} {'err':>7}")
    for bs in [4,16,64,128]:
        h=torch.randn(bs,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.1
        ti=torch.randint(0,E,(bs,TOPK),dtype=torch.int32,device=DEV)
        tw=torch.rand(bs,TOPK,dtype=torch.bfloat16,device=DEV)*0.1+0.9
        w13_bf16=torch.randn(E,N2,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.02
        w2_bf16=torch.randn(E,K_HIDDEN,N_INTER,dtype=torch.bfloat16,device=DEV)*0.02
        w13f,w13s=per_block_quant_fp8(w13_bf16); w2f,w2s=per_block_quant_fp8(w2_bf16)
        BM=16 if bs<=32 else 32
        out_f=moe_fused(h,w13f,w13s,w2f,w2s,tw,ti,BM)
        t_f=bench_fn(lambda: moe_fused(h,w13f,w13s,w2f,w2s,tw,ti,BM))
        def prod(): return fused_moe(hidden_states=h,w1=w13f,w2=w2f,topk_weight=tw.to(torch.float32),topk_ids=ti,quant_type=QuantType.per_128x128,activation=ActivationType.Silu,w1_scale=w13s,w2_scale=w2s,a1_scale=None,a2_scale=None)
        out_p=prod(); t_p=bench_fn(prod)
        err=(out_f.to(torch.float32)-out_p.to(torch.float32)).abs().max().item()
        print(f"{bs:>5} {BM:>4} {t_f:>9.1f} {t_p:>9.1f} {t_p/t_f:>6.2f}x {err:>7.3f}")

if __name__=="__main__": main()
