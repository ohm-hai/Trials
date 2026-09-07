"""Phase 8c single: confirm megakernel wins + correct at bs=64 boundary."""
import torch, sys, os
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL"]="1"
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL_MIN_TOKENS"]="64"
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS"]="1024"
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from aiter.fused_moe import fused_moe, QuantType, ActivationType
import sglang.srt.layers.moe.moe_runner.aiter as A

def bench_fn(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=torch.cuda.Event(enable_timing=True); t1=torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1)*1e3/iters

def main():
    torch.manual_seed(0)
    bs=64
    h=torch.randn(bs,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.1
    ti=torch.randint(0,E,(bs,TOPK),dtype=torch.int32,device=DEV)
    tw=torch.rand(bs,TOPK,dtype=torch.bfloat16,device=DEV)*0.1+0.9
    w13_bf16=torch.randn(E,N2,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.02
    w2_bf16=torch.randn(E,K_HIDDEN,N_INTER,dtype=torch.bfloat16,device=DEV)*0.02
    w13f,w13s=per_block_quant_fp8(w13_bf16); w2f,w2s=per_block_quant_fp8(w2_bf16)
    def prod(): return fused_moe(hidden_states=h,w1=w13f,w2=w2f,topk_weight=tw.to(torch.float32),topk_ids=ti,quant_type=QuantType.per_128x128,activation=ActivationType.Silu,w1_scale=w13s,w2_scale=w2s,a1_scale=None,a2_scale=None)
    out_p=prod(); t_p=bench_fn(prod)
    out_mk=A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti)
    t_mk=bench_fn(lambda: A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti))
    err=(out_mk.to(torch.float32)-out_p.to(torch.float32)).abs().max().item()
    print(f"bs=64  mega={t_mk:.1f}us  prod={t_p:.1f}us  spd={t_p/t_mk:.2f}x  err={err:.3f}")
    # also bs=128
    bs=128
    h=torch.randn(bs,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.1
    ti=torch.randint(0,E,(bs,TOPK),dtype=torch.int32,device=DEV)
    tw=torch.rand(bs,TOPK,dtype=torch.bfloat16,device=DEV)*0.1+0.9
    w13f,w13s=per_block_quant_fp8(torch.randn(E,N2,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.02)
    w2f,w2s=per_block_quant_fp8(torch.randn(E,K_HIDDEN,N_INTER,dtype=torch.bfloat16,device=DEV)*0.02)
    def prod2(): return fused_moe(hidden_states=h,w1=w13f,w2=w2f,topk_weight=tw.to(torch.float32),topk_ids=ti,quant_type=QuantType.per_128x128,activation=ActivationType.Silu,w1_scale=w13s,w2_scale=w2s,a1_scale=None,a2_scale=None)
    out_p=prod2(); t_p=bench_fn(prod2)
    out_mk=A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti)
    t_mk=bench_fn(lambda: A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti))
    err=(out_mk.to(torch.float32)-out_p.to(torch.float32)).abs().max().item()
    print(f"bs=128 mega={t_mk:.1f}us  prod={t_p:.1f}us  spd={t_p/t_mk:.2f}x  err={err:.3f}")

if __name__=="__main__": main()
