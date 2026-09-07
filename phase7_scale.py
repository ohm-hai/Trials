"""Phase 7: scale the decode megakernel (BLOCK_M=128) to serving-level decode
batches and beyond, to confirm the win holds and find where it inverts
(prefill regime). Reports speedup vs round-robin at each batch."""
import torch, sys
sys.path.insert(0, "/root/moe_bench_results")
from phase3_blockscale import _launch, make_blockwise_fp8, build_routing_balanced
from phase6_tune import bench, E, TOPK, K, N, DEV

BM, BN, BK, NX = 128, 128, 128, 8

def main():
    print("=== Phase 7: scale decode megakernel (BLOCK_M=128) vs round-robin ===")
    print(f"{'batch':>7} {'npm':>5} {'round-robin us':>14} {'megakernel us':>14} {'spd':>7} {'per-XCD wt MB':>11}")
    for bs in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        a_fp8, a_scale, b_fp8, b_scale = make_blockwise_fp8(bs, E, K, N, DEV, BN, BK)
        sti, eid, npm, nvt = build_routing_balanced(bs, E, TOPK, DEV)
        npp_t = torch.tensor([nvt], dtype=torch.int32, device=DEV)
        # round-robin
        c_rr = torch.zeros(npm * BM, N, dtype=torch.bfloat16, device=DEV)
        for _ in range(10): _launch(a_fp8, b_fp8, a_scale, b_scale, c_rr, sti, eid, npp_t, nvt, npm, 0)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(40): _launch(a_fp8, b_fp8, a_scale, b_scale, c_rr, sti, eid, npp_t, nvt, npm, 0)
        t1.record(); torch.cuda.synchronize()
        t_rr = t0.elapsed_time(t1) * 1e3 / 40
        # megakernel
        t_mk = bench(bs, BM, BN, BK, NX, warmup=10, iters=40)
        wt_mb = K * (N // 8) / 1e6
        print(f"{bs:>7} {npm:>5} {t_rr:>14.1f} {t_mk:>14.1f} {t_rr/t_mk:>6.2f}x {wt_mb:>11.3f}")

if __name__ == "__main__":
    main()
