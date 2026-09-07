"""Phase B validation: aiter FlashKDA vs sglang Triton chunk_kda (KDA prefill)
on the Kimi-K3 KDA shape (H=HV=12, K=V=128, safe-gate lower_bound=-5.0).

This exercises the kernel the `AiterFlashKDAKernel` adapter calls
(`aiter.ops.triton.kimi_delta_attn.chunk_kimi_delta_attn`, which auto-selects the
two-kernel FlashKDA path for K=V=128 / no-GVA / fused-gate / in-kernel l2norm +
beta-sigmoid / safe-gate) against sglang's existing Triton `chunk_kda` fallback.

Varlen, single packed sequence (B=1), cu_seqlens=[0,T]. State layout:
sglang pool [N,H,V,K]; aiter called with state_v_first=True -> [N,HV,V,K] (== sglang).
"""
import os
import sys
import time
import statistics

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/scratch/aiter")

from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from sglang.kernels.ops.attention.fla.kda import chunk_kda

DEV = torch.device("cuda:0")
H = 12          # Kimi-K3 num_qk_heads == num_v_heads (no GVA)
K = 128         # head_k_dim
V = 128         # head_v_dim
LOWER_BOUND = -5.0
SCALE = K**-0.5
DTYPE = torch.bfloat16


def make_inputs(T, N=1, seed=0):
    g = torch.manual_seed(seed)
    q = torch.randn(1, T, H, K, dtype=DTYPE, device=DEV)
    k = torch.randn(1, T, H, K, dtype=DTYPE, device=DEV)
    v = torch.randn(1, T, H, V, dtype=DTYPE, device=DEV)
    gate = torch.randn(1, T, H, K, dtype=DTYPE, device=DEV)   # RAW pre-activation gate
    beta = torch.randn(1, T, H, dtype=torch.float32, device=DEV)  # RAW logits
    A_log = torch.randn(H, dtype=torch.float32, device=DEV)
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=DEV)
    # sglang pool state [N, H, V, K]
    ssm_states = torch.zeros(N, H, V, K, dtype=torch.float32, device=DEV)
    cache_indices = torch.zeros(N, dtype=torch.int32, device=DEV)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    return q, k, v, gate, beta, A_log, dt_bias, ssm_states, cache_indices, cu_seqlens


def bench_aiter(T, iters=30, warmup=5):
    q, k, v, g, beta, A_log, dt_bias, ssm_states, ci, cu = make_inputs(T)
    init = ssm_states[ci].contiguous()  # [1, H, V, K] (state_v_first=True)
    for _ in range(warmup):
        o, fs = chunk_kimi_delta_attn(
            q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, scale=SCALE,
            initial_state=init, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True,
            lower_bound=LOWER_BOUND, state_v_first=True, cu_seqlens=cu,
        )
    torch.cuda.synchronize(DEV)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(DEV); t0 = time.perf_counter()
        o, fs = chunk_kimi_delta_attn(
            q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, scale=SCALE,
            initial_state=init, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True,
            lower_bound=LOWER_BOUND, state_v_first=True, cu_seqlens=cu,
        )
        torch.cuda.synchronize(DEV); ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), o, fs


def bench_triton(T, iters=30, warmup=5):
    q, k, v, g, beta, A_log, dt_bias, ssm_states, ci, cu = make_inputs(T)
    cu64 = cu.to(torch.int64)
    for _ in range(warmup):
        ssm_states.zero_()
        o = chunk_kda(
            q, k, v, g, beta, scale=SCALE, initial_state=ssm_states,
            initial_state_indices=ci, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu64, A_log=A_log, dt_bias=dt_bias,
            lower_bound=LOWER_BOUND, output_intermediate_states=False,
            beta_is_raw=True,
        )
    torch.cuda.synchronize(DEV)
    ts = []
    for _ in range(iters):
        ssm_states.zero_()
        torch.cuda.synchronize(DEV); t0 = time.perf_counter()
        o = chunk_kda(
            q, k, v, g, beta, scale=SCALE, initial_state=ssm_states,
            initial_state_indices=ci, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu64, A_log=A_log, dt_bias=dt_bias,
            lower_bound=LOWER_BOUND, output_intermediate_states=False,
            beta_is_raw=True,
        )
        torch.cuda.synchronize(DEV); ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), o


def main():
    print(f"KDA prefill: aiter FlashKDA vs sglang Triton chunk_kda (Kimi-K3: H={H} K={K} V={V} lb={LOWER_BOUND})")
    print(f"{'T':>6} {'aiter_us':>10} {'triton_us':>10} {'speedup':>8} {'allclose':>9}")
    rows = []
    for T in [512, 1024, 2048, 4096, 8192]:
        a_med, a_o, a_fs = bench_aiter(T)
        t_med, t_o = bench_triton(T)
        # correctness: both bf16, l2norm+fused gate -> tolerant compare
        ok = torch.allclose(a_o.float(), t_o.float(), atol=2e-2, rtol=2e-2)
        sp = t_med / a_med
        print(f"{T:>6} {a_med:>10.2f} {t_med:>10.2f} {sp:>8.2f}x {str(ok):>9}")
        rows.append((T, a_med, t_med, sp, ok))

    out = "/root/moe_bench_results/micro/kda_prefill_aiter_vs_triton.csv"
    with open(out, "w") as f:
        f.write("# KDA prefill: aiter FlashKDA vs sglang Triton chunk_kda (gfx950, MI355X VF, GPU0)\n")
        f.write("# Kimi-K3 shape: H=HV=12 K=V=128 safe-gate lower_bound=-5.0 varlen B=1\n")
        f.write("# aiter = chunk_kimi_delta_attn (state_v_first=True, FlashKDA two-kernel path)\n")
        f.write("# triton = sglang kernels/ops/attention/fla/kda.chunk_kda (beta_is_raw=True)\n")
        f.write("T,aiter_us,triton_us,speedup,allclose\n")
        for T, a, t, sp, ok in rows:
            f.write(f"{T},{a:.2f},{t:.2f},{sp:.2f},{ok}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
