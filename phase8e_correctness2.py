"""Phase 8e correctness v2: runtime-NPM orchestrator vs bf16 reference (no prod hsaco)."""
import torch, sys, os
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL"]="1"
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL_MIN_TOKENS"]="64"
os.environ["SGLANG_MOE_DECODE_MEGAKERNEL_MAX_TOKENS"]="1024"
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
import sglang.srt.layers.moe.moe_runner.aiter as A

def bf16_ref(h, w13, w2, ti, tw):
    M, K = h.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=h.device)
    for t in range(M):
        acc = torch.zeros(K, dtype=torch.float32, device=h.device)
        for j in range(TOPK):
            e = int(ti[t, j])
            g = h[t] @ w13[e].t()
            inter = torch.nn.functional.silu(g[0::2]) * g[1::2]
            acc += (inter @ w2[e].t()) * float(tw[t, j])
        out[t] = acc
    return out.to(torch.bfloat16)

def main():
    torch.manual_seed(0)
    print("=== Phase 8e correctness v2: runtime-NPM orchestrator vs bf16 ref ===")
    print(f"{'bs':>5} {'err':>8}")
    for bs in [64, 128, 256]:
        h=torch.randn(bs,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.1
        ti=torch.randint(0,E,(bs,TOPK),dtype=torch.int32,device=DEV)
        tw=torch.rand(bs,TOPK,dtype=torch.bfloat16,device=DEV)*0.1+0.9
        w13_bf16=torch.randn(E,N2,K_HIDDEN,dtype=torch.bfloat16,device=DEV)*0.02
        w2_bf16=torch.randn(E,K_HIDDEN,N_INTER,dtype=torch.bfloat16,device=DEV)*0.02
        w13f,w13s=per_block_quant_fp8(w13_bf16); w2f,w2s=per_block_quant_fp8(w2_bf16)
        out_m=A._moe_decode_megakernel_full(h,w13f,w13s,w2f,w2s,tw.to(torch.float32),ti)
        ref=bf16_ref(h,w13_bf16,w2_bf16,ti,tw)
        err=(out_m.to(torch.float32)-ref.to(torch.float32)).abs().max().item()
        print(f"{bs:>5} {err:>8.3f}")

if __name__=="__main__": main()
