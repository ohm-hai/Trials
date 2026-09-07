"""Phase 13: re-validate the decode win on the FIXED integrated orchestrator
(_moe_decode_megakernel_full, with A_BY_SORTED stage-2 + correct index_add_
unpermute) vs prod aiter.fused_moe hsaco. The old Phase 7 win (1.26-3.60x)
was timed on the buggy stage-2 path; this measures the honest post-fix win
on CORRECT output (verified vs bf16 ref)."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from aiter.fused_moe import fused_moe, QuantType, ActivationType
from sglang.srt.layers.moe.moe_runner.aiter import _moe_decode_megakernel_full


def bench_fn(fn, warmup=10, iters=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def bf16_ref(h, w13, w2, ti, tw):
    M, K = h.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    for t in range(M):
        for k in range(TOPK):
            e = int(ti[t, k].item())
            g = w13[e, 0::2].to(torch.float32); u = w13[e, 1::2].to(torch.float32)
            inter = torch.nn.functional.silu(h[t].to(torch.float32) @ g.T) * (h[t].to(torch.float32) @ u.T)
            out[t] += (inter @ w2[e].to(torch.float32).T) * tw[t, k].to(torch.float32)
    return out.to(torch.bfloat16)


def main():
    torch.manual_seed(0)
    w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
    w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
    w13f, w13s = per_block_quant_fp8(w13_bf16)
    w2f, w2s = per_block_quant_fp8(w2_bf16)
    print("=== Phase 13: decode win re-validation (FIXED orchestrator) ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER}")
    # Correctness vs bf16 ref at bs=64 (confirm npm fix + A_BY_SORTED fix => correct output)
    bs = 64
    h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
    ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
    tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
    ref = bf16_ref(h, w13_bf16, w2_bf16, ti, tw)
    mk_out = _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw, ti)
    rel = (mk_out - ref).abs().max().item() / ref.abs().max().item()
    print(f"correctness bs=64: rel_err={rel:.4f} (want <<1.0; 1.0 = wrong)\n")
    print(f"{'batch':>7} {'megakernel us':>14} {'prod fused_moe us':>18} {'spd':>7}")
    for bs in [64, 128, 256, 512, 1024]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        def mk(): return _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw, ti)
        def prod():
            return fused_moe(hidden_states=h, w1=w13f, w2=w2f, topk_weight=tw.to(torch.float32),
                               topk_ids=ti, quant_type=QuantType.per_128x128,
                               activation=ActivationType.Silu,
                               w1_scale=w13s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                               gate_mode="interleave")
        mk(); print("  mk warmup done", flush=True); prod(); print("  prod warmup done", flush=True)  # JIT warmup
        t_mk = bench_fn(mk); t_p = bench_fn(prod)
        print(f"{bs:>7} {t_mk:>14.1f} {t_p:>18.1f} {t_p/t_mk:>6.2f}x", flush=True)


if __name__ == "__main__":
    main()
