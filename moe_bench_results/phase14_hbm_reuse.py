"""Phase 14: measure HBM reuse of the PRODUCTION aiter.fused_moe hsaco at decode.

For each bs in {64, 256, 1024}:
  - set up GLM-5.2-sized FP8 weights + fixed-seed routing
  - print the EXACT active (distinct) expert count
  - run prod fused_moe (hsaco) for WARMUP + ITERS iterations in a tight loop

Run under `rocprofv2 -i <pmc_file>` (FETCH_SIZE/WRITE_SIZE); parse the fmoe
kernel HBM fetch, divide by the fmoe dispatch count, and compare to the
weight-traffic floor = active_experts * (w13 + w2 per expert) [fp8 bytes].

  redundancy = actual_per_dispatch / floor
    ~1.0  -> hsaco already exploits LLC (each expert loaded once) -> lever C moot
    >>1.0 -> reuse to capture (experts loaded multiple times) -> lever C viable
"""
import torch, sys, time
sys.path.insert(0, "/scratch/aiter")
sys.path.insert(0, "/scratch/sglang/python")
sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from aiter.fused_moe import fused_moe, QuantType, ActivationType

WARMUP = 5
ITERS = 20

torch.manual_seed(42)
# Fixed weights (same for all batch sizes)
w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
w13f, w13s = per_block_quant_fp8(w13_bf16)
w2f, w2s = per_block_quant_fp8(w2_bf16)

# Per-expert weight bytes (fp8): w13 = 2*N_inter*K, w2 = K*N_inter
W13_PER = 2 * N_INTER * K_HIDDEN      # bytes (fp8 = 1 byte/element)
W2_PER = K_HIDDEN * N_INTER
PER_EXPERT = W13_PER + W2_PER

def run(bs):
    # Fixed-seed routing per bs
    g = torch.Generator(device=DEV).manual_seed(1000 + bs)
    h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV, generator=g) * 0.1
    ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV, generator=g)
    tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV, generator=g) * 0.1 + 0.9
    active = int(ti.unique().numel())
    floor = active * PER_EXPERT
    print(f"=== bs={bs} active_experts={active} floor={floor/1e6:.2f}MB "
          f"({floor/1e9:.3f}GB) per_expert={PER_EXPERT/1e6:.2f}MB ===", flush=True)

    def prod():
        return fused_moe(hidden_states=h, w1=w13f, w2=w2f,
                          topk_weight=tw.to(torch.float32), topk_ids=ti,
                          quant_type=QuantType.per_128x128,
                          activation=ActivationType.Silu,
                          w1_scale=w13s, w2_scale=w2s,
                          a1_scale=None, a2_scale=None, gate_mode="interleave")
    # warmup (also profiled by rocprof, accounted for via dispatch count)
    for _ in range(WARMUP):
        prod()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        prod()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"--- bs={bs} done: {ITERS} iters in {dt:.3f}s ({dt/ITERS*1000:.2f}ms/iter) "
          f"total_dispatches={WARMUP+ITERS} ===", flush=True)

print(f"GLM-5.2 FP8: E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER} N2={N2}", flush=True)
print(f"per-expert weight: w13={W13_PER/1e6:.2f}MB w2={W2_PER/1e6:.2f}MB "
      f"total={PER_EXPERT/1e6:.2f}MB", flush=True)
BSS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [64, 256, 1024]
for bs in BSS:
    run(bs)
print("PHASE14_DONE", flush=True)
