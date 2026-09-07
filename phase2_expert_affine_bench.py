#!/usr/bin/env python
"""Phase 2 — expert-affine vs round-robin remap_xcd on the Triton fused-MoE stage1 GEMM.

Self-contained microbench: a minimal Triton fused-MoE stage1 (gate/up) GEMM with a
REMAP_MODE constexpr:
  0 = round-robin remap_xcd (aiter original, xcd = pid % NUM_XCDS)
  1 = expert-affine: all tiles of expert e -> XCD(e % NUM_XCDS), via a
      host-precomputed pid_remap table (one extra tl.load in the kernel).

GLM-5.2 stage1 shape: M_tokens routed rows (M_tokens*topk), E=256, K=6144, N=2048, bf16.
Correctness vs torch reference; latency timing. The locality lever is
isolated: same kernel, only the remap differs -> L2-hit delta is purely locality.
"""
import os, time
import torch
import triton
import triton.language as tl

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
HIP = torch.version.hip is not None
NUM_XCD = 8
E = 256
TOPK = 8
K = 6144          # hidden
N = 2048          # intermediate (gate/up output)
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
GROUP_M = 8


@triton.jit
def _fused_moe_stage1_kernel(
    a_ptr, b_ptr, c_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, pid_remap_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    N, K, EM,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, REMAP_MODE: tl.constexpr, NUM_XCDS: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    GRID_MN = num_pid_n * num_pid_m
    if pid >= GRID_MN:
        return

    if REMAP_MODE == 1:
        # expert-affine: launch pid controls XCD (pid % NUM_XCDS == target_xcd),
        # but data tile is decoupled via a tile_id table.
        tile_id = tl.load(pid_remap_ptr + pid.to(tl.int64))        # raw tile index
    else:
        # round-robin (aiter original): launch pid == data pid
        tile_id = pid

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    offs_token_id = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_token = tl.where(token_mask, offs_token, 0)
    off_experts = tl.load(expert_ids_ptr + pid_m.to(tl.int64)).to(tl.int64)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // TOPK * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=token_mask[:, None])


def build_routing(M_tokens, E, topk, device, seed=0, balanced=True):
    """Build sorted_token_ids, expert_ids for M_tokens routed rows (M_tokens*topk).
    balanced=True: each token picks exactly one expert per XCD (8 XCDs, topk=8) -> each
      XCD gets exactly M_tokens routed tokens -> 128 tiles/XCD (no imbalance).
    balanced=False: random topk distinct experts (skewed, for stress-testing)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.empty(M_tokens, topk, dtype=torch.int32)
    if balanced and topk == NUM_XCD:
        # one expert per XCD: experts grouped as [e//32 for e in 0..255] -> xcd = e//32
        for i in range(M_tokens):
            per_xcd = [torch.randperm(32, generator=g)[0] + xcd * 32 for xcd in range(NUM_XCD)]
            idx[i] = torch.tensor(per_xcd, dtype=torch.int32)
    else:
        for i in range(M_tokens):
            idx[i] = torch.randperm(E, generator=g)[:topk].to(torch.int32)
    flat_idx = idx.reshape(-1)  # (M_tokens*topk,)
    # sort by expert -> sorted_token_ids (FLAT index in [0, M*topk)) and expert_ids per block-row
    # flat index i -> token = i // topk  (vLLM/aiter convention: kernel does offs_token // top_k)
    flat_token = torch.arange(M_tokens * topk, dtype=torch.int32)
    order = torch.argsort(flat_idx, stable=True)
    sorted_expert = flat_idx[order].to(torch.int32)
    sorted_token = flat_token[order].to(torch.int32).to(device)
    num_pid_m = (M_tokens * topk + BLOCK_M - 1) // BLOCK_M
    expert_ids = torch.empty(num_pid_m, dtype=torch.int32, device=device)
    for pm in range(num_pid_m):
        pos = pm * BLOCK_M
        expert_ids[pm] = sorted_expert[pos] if pos < len(sorted_expert) else -1
    pad = num_pid_m * BLOCK_M - len(sorted_token)
    if pad > 0:
        sorted_token = torch.cat([sorted_token, torch.full((pad,), M_tokens * topk, dtype=torch.int32, device=device)])
    return sorted_token.to(device), expert_ids, num_pid_m, (M_tokens * topk)


def build_pid_remap(expert_ids, num_pid_m, num_pid_n, NUM_XCD):
    """expert-affine remap. Returns pid_remap[launch_pid] -> raw_tile.
    Launch grid is reordered so launch_pid % NUM_XCD == target_xcd (expert of the tile),
    i.e. for xcd X the 128 launch pids are {X + k*NUM_XCDS : k=0..127}.
    Data indexing is decoupled: kernel loads tile_id = pid_remap[launch_pid]."""
    GRID_MN = num_pid_m * num_pid_n
    pids_per_xcd = (GRID_MN + NUM_XCD - 1) // NUM_XCD  # 128 for GLM-5.2
    # group raw tiles by target xcd
    groups = [[] for _ in range(NUM_XCD)]
    for raw in range(GRID_MN):
        pid_m = raw // num_pid_n
        exp = expert_ids[pid_m] if pid_m < len(expert_ids) else -1
        tx = (exp % NUM_XCD) if exp >= 0 else 0
        groups[tx].append(raw)
    pid_remap = torch.full((GRID_MN,), -1, dtype=torch.int32)
    overflow = list(range(GRID_MN))
    for xcd in range(NUM_XCD):
        g = groups[xcd]
        for k, raw in enumerate(g):
            if k < pids_per_xcd:
                launch_pid = xcd + k * NUM_XCD
                pid_remap[launch_pid] = raw
                overflow.remove(raw)
    free = [i for i in range(GRID_MN) if pid_remap[i] < 0]
    for raw in overflow:
        if free:
            pid_remap[free.pop()] = raw
    return pid_remap


def torch_ref(a, b, sorted_token, expert_ids, topk, M_tokens):
    """torch reference: out[i] = a[sorted_token[i]] @ b[expert_ids[block(i)]]"""
    M_pad = sorted_token.shape[0]
    out = torch.zeros(M_pad, N, dtype=torch.bfloat16, device=a.device)
    for pm in range(M_pad // BLOCK_M):
        pos = pm * BLOCK_M
        exp = int(expert_ids[pm])
        if exp < 0:
            continue
        toks = sorted_token[pos:pos + BLOCK_M]
        toks = toks[toks < M_tokens * topk]
        a_block = a[toks // topk].to(torch.float32)  # (blk, K)
        b_block = b[exp].to(torch.float32)        # (K, N) -- b is (E, K, N)
        out_block = (a_block @ b_block).to(torch.bfloat16)  # (blk, N)
        for j, t in enumerate(toks):
            out[t] = out_block[j]
    return out


def bench(M_tokens, mode, warmup=20, iters=50):
    device = DEV
    a = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(E, K, N, dtype=torch.bfloat16, device=device) * 0.02
    sorted_token, expert_ids, num_pid_m, npp = build_routing(M_tokens, E, TOPK, device, balanced=True)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_remap = build_pid_remap(expert_ids, num_pid_m, num_pid_n, NUM_XCD).to(device)
    c = torch.empty(num_pid_m * BLOCK_M, N, dtype=torch.bfloat16, device=device)
    grid = (num_pid_m * num_pid_n,)
    args = (a, b, c, sorted_token, expert_ids, pid_remap,
            torch.tensor([npp], dtype=torch.int32, device=device),
            M_tokens * TOPK,
            N, K, E,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, mode, NUM_XCD, TOPK)
    for _ in range(warmup):
        _fused_moe_stage1_kernel[grid](*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _fused_moe_stage1_kernel[grid](*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us/launch


def main():
    torch.manual_seed(0)
    print(f"=== Phase 2: expert-affine vs round-robin remap_xcd (Triton fused-MoE stage1) ===")
    print(f"shape: E={E} topk={TOPK} K={K} N={N} bf16 | NUM_XCD={NUM_XCD} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}")
    # correctness: compare round-robin (mode 0) vs torch ref at M_tokens=2048
    M_tokens = 2048
    a = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(E, K, N, dtype=torch.bfloat16, device=DEV) * 0.02
    sorted_token, expert_ids, num_pid_m, npp = build_routing(M_tokens, E, TOPK, DEV, balanced=True)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_remap = build_pid_remap(expert_ids, num_pid_m, num_pid_n, NUM_XCD).to(DEV)
    c_rr = torch.empty(num_pid_m * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
    c_ea = torch.empty(num_pid_m * BLOCK_M, N, dtype=torch.bfloat16, device=DEV)
    grid = (num_pid_m * num_pid_n,)
    print(f"diag: num_pid_m={num_pid_m} num_pid_n={num_pid_n} grid={grid} npp={npp}")
    print(f"diag: sorted_token range [{sorted_token.min().item()},{sorted_token.max().item()}] len={sorted_token.numel()}")
    print(f"diag: expert_ids range [{expert_ids.min().item()},{expert_ids.max().item()}] len={expert_ids.numel()} any_neg={(expert_ids<0).sum().item()}")
    print(f"diag: pid_remap range [{pid_remap.min().item()},{pid_remap.max().item()}] any_neg={(pid_remap<0).sum().item()}")
    npp_t = torch.tensor([npp], dtype=torch.int32, device=DEV)
    nvt = M_tokens * TOPK
    # kernel sig: a_ptr, b_ptr, c_ptr, sti, eid, pid_remap, npp_ptr, nvt, N, K, EM,
    #             sam, sak, sbe, sbk, sbn, scm, scn,
    #             BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, REMAP_MODE, NUM_XCDS, TOPK
    def call(c_out, mode):
        _fused_moe_stage1_kernel[grid](
            a, b, c_out, sorted_token, expert_ids, pid_remap, npp_t, nvt,
            N, K, E, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
            c_out.stride(0), c_out.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, mode, NUM_XCD, TOPK)
    print("launching mode 0 (round-robin)...", flush=True)
    call(c_rr, 0)
    torch.cuda.synchronize()
    print("mode 0 done.", flush=True)
    print("launching mode 1 (expert-affine)...", flush=True)
    call(c_ea, 1)
    torch.cuda.synchronize()
    print("mode 1 done.", flush=True)
    ref = torch_ref(a, b, sorted_token, expert_ids, TOPK, M_tokens).to(DEV)
    # c is in SORTED order (c[i] = result for sorted_token[i]); ref is in FLAT order (ref[t]=result for token t).
    # gather ref by sorted_token to align, then compare over valid (non-pad) rows.
    valid = sorted_token < M_tokens * TOPK
    sti_clamped = sorted_token.clamp(max=ref.shape[0] - 1)
    ref_sorted = ref[sti_clamped]
    diff_rr = (c_rr[valid].to(torch.float32) - ref_sorted[valid].to(torch.float32)).abs().max().item()
    diff_ea = (c_ea[valid].to(torch.float32) - ref_sorted[valid].to(torch.float32)).abs().max().item()
    print(f"correctness @ M_tokens={M_tokens}: max_abs_err round-robin={diff_rr:.4f}  expert-affine={diff_ea:.4f}  (vs torch ref)")
    # cross-check the two remaps produce identical results
    diff_cross = (c_rr[valid].to(torch.float32) - c_ea[valid].to(torch.float32)).abs().max().item()
    print(f"round-robin vs expert-affine max_abs_diff={diff_cross:.4f} (should be ~0)")
    # latency sweep (M_tokens >= 2048 so per-expert tokens >= BLOCK_M=64 -> no padding inflation -> balanced XCDs)
    print(f"\n{'M_tokens':>10} {'M(topk)':>10} {'round-robin us':>14} {'expert-affine us':>16} {'speedup':>9}")
    for M_tokens in [2048, 4096, 8192, 16384]:
        t_rr = bench(M_tokens, 0)
        t_ea = bench(M_tokens, 1)
        sp = t_rr / t_ea if t_ea > 0 else 0
        print(f"{M_tokens:>10} {M_tokens*TOPK:>10} {t_rr:>14.2f} {t_ea:>16.2f} {sp:>8.2f}x")


if __name__ == "__main__":
    main()
