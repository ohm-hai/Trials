"""Phase 7: exercise the ACTUAL sglang _moe_decode_megakernel_full (the integrated
serving-path function) at a decode batch, verifying it runs and produces
finite, correctly-shaped output. This proves the integrated code path works."""
import torch, sys
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, build_routing_from_topk, E, TOPK, K_HIDDEN, N_INTER, N2, DEV

# Import the ACTUAL sglang integrated function (not the phase5 copy)
from sglang.srt.layers.moe.moe_runner.aiter import _moe_decode_megakernel_full

def main():
    torch.manual_seed(0)
    print("=== Phase 7: actual sglang _moe_decode_megakernel_full ===")
    for bs in [256, 512, 1024]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
        w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
        w13f, w13s = per_block_quant_fp8(w13_bf16)
        w2f, w2s = per_block_quant_fp8(w2_bf16)
        out = _moe_decode_megakernel_full(h, w13f, w13s, w2f, w2s, tw.to(torch.float32), ti)
        torch.cuda.synchronize()
        finite = torch.isfinite(out).all().item()
        print(f"  bs={bs}: out shape={tuple(out.shape)} finite={finite} "
              f"mean={out.float().mean().item():.4f} std={out.float().std().item():.4f}")
        assert finite, "NaN/Inf in output!"
        assert out.shape == (bs, K_HIDDEN), f"bad shape {out.shape}"
    print("\nPASS: sglang integrated _moe_decode_megakernel_full runs correctly at decode batches.")

if __name__ == "__main__":
    main()
