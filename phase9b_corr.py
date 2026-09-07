"""Phase 9b-corr: verify the integrated megakernel orchestrator vs a bf16
per-token-per-expert reference (both use INTERLEAVED gate/up). Small M."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.aiter import _moe_decode_megakernel_full

def bf16_ref(h, w13, w2, ti, tw):
    M, K = h.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    for t in range(M):
        for k in range(TOPK):
            e = int(ti[t, k].item())
            g = w13[e, 0::2].to(torch.float32)   # interleaved gate
            u = w13[e, 1::2].to(torch.float32)   # interleaved up
            inter = torch.nn.functional.silu(h[t].to(torch.float32) @ g.T) * (h[t].to(torch.float32) @ u.T)
            out[t] += (inter @ w2[e].to(torch.float32).T) * tw[t, k].to(torch.float32)
    return out.to(torch.bfloat16)

def main():
    torch.manual_seed(0)
    for M in [64, 128, 256]:
        h = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        w13 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13)
        w2f, w2s = per_block_quant_fp8(w2)
        out_mk = _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw, ti)
        out_ref = bf16_ref(h, w13, w2, ti, tw)
        err = (out_mk.to(torch.float32) - out_ref.to(torch.float32)).abs().max().item()
        rel = err / (out_ref.float().abs().max().item() + 1e-6)
        print(f"M={M:>5} max_abs_err={err:.4f} out_mag={out_ref.float().abs().max().item():.4f} rel_err={rel:.3f}")

if __name__ == "__main__":
    main()
