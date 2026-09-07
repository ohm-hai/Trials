"""Phase 8e correctness: runtime-NPM base megakernel orchestrator vs prod hsaco."""
import torch, sys, os
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL"]="1"
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL_MIN_TOKENS"]="64"
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS"]="1024"
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from aiter.fused_moe import fused_moe, QuantType, ActivationType
import sglang.srt.layers.moe.moe_runner.aiter as A

def main():
    torch.manual_seed(0)
    print("=== Phase 8e correctness: runtime-NPM orchestrator vs prod ===")
    print(f"{'bs':>5} {'npm':>5} {'mk us':>8} {'prod us':>9} {'spd':>7} {'err':>7}")
    for bs in [64, 128, 256]:
        h=torch.randn(bs,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.1
        ti=torch.randint(0,E,(bs,TOPK),dtype=torch.int32,device=DEV)
        tw=torch.rand(bs,TOPK,dtype=torch.bfloat16,device=DEV)*0.1+0.9
        w13f,w13s=per_block_quant_fp8(torch.randn(E,N2,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.02)
        w2f,w2s=per_block_quant_fp8(torch.randn(E,K_HIDDEN,N_INTER,dtype=torch.bfloat16,device=DEV)*0.02)
        def prod(): return fused_moe(hidden_states=h,w1=w13f,w2=w2f,topk_weight=tw.to(torch.float32),topk_ids=ti,quant_type=QuantType.per_128x128,activation=ActivationType.Silu,w1_scale=w13s,w2_scale=w2s,a1_scale=None,a2_scale=None)
        out_p=prod()
        # warm
        for _ in range(3): prod(); A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti)
        torch.cuda.synchronize()
        t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(30): prod()
        t1.record(); torch.cuda.synchronize(); tp=t0.elapsed_time(t1)*1e3/30
        t0.record()
        for _ in range(30): A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti)
        t1.record(); torch.cuda.synchronize(); tm=t0.elapsed_time(t1)*1e3/30
        out_m=A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti)
        err=(out_m.to(torch.float32)-out_p.to(torch.float32)).abs().max().item()
        print(f"{bs:>5} {'?':>5} {tm:>8.1f} {tp:>9.1f} {tp/tm:>6.2f}x {err:>7.3f}")

if __name__=="__main__": main()
