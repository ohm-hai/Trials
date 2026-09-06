"""Phase 6: tune BLOCK_M / N-slice for the decode megakernel.
Sweep BLOCK_M in {16,32,64,128} and NUM_XCD in {4,8,16} (N-slice granularity)
to find the best decode config. Reports correctness (vs round-robin) and speedup
at decode batch 512/1024/2048."""
import torch, triton, triton.language as tl
import sys
sys.path.insert(0, "/scratch/aiter")
from phase3_blockscale import _mega_bs_kernel as _moe_decode_mega_kernel, _launch

DEV = "cuda"
E = 256; TOPK = 8; K = 6144; N = 2048
GROUP_N = 128; GROUP_K = 128


def build_routing_balanced(M_tokens, E, topk, device, BM):
    g = torch.Generator(device="cpu").manual_seed(7)
    idx = torch.empty(M_tokens, topk, dtype=torch.int32)
    NX = 8
    for i in range(M_tokens):
        idx[i] = torch.tensor([torch.randperm(32, generator=g)[0].item() + xcd * 32 for xcd in range(NX)], dtype=torch.int32)
    flat_idx = idx.reshape(-1)
    flat_token = torch.arange(M_tokens * topk, dtype=torch.int32)
    order = torch.argsort(flat_idx, stable=True)
    sorted_expert = flat_idx[order].to(torch.int32)
    sorted_token = flat_token[order].to(torch.int32).to(device)
    npm = (M_tokens * topk + BM - 1) // BM
    eid = torch.empty(npm, dtype=torch.int32, device=device)
    for pm in range(npm):
        pos = pm * BM
        eid[pm] = sorted_expert[pos] if pos < len(sorted_expert) else 0
    pad = npm * BM - len(sorted_token)
    if pad > 0:
        sorted_token = torch.cat([sorted_token, torch.full((pad,), M_tokens * topk, dtype=torch.int32, device=device)])
    return sorted_token, eid, npm, M_tokens * topk


def make_blockwise_fp8(M_tokens, E, K, N, device, bn=128, bk=128):
    a_bf16 = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(E, K, N, dtype=torch.bfloat16, device=device) * 0.02
    a_max = a_bf16.abs().reshape(M_tokens, K // bk, bk).amax(dim=-1).float()
    a_scale = (a_max / 448.0).clamp(min=1e-12)
    a_fp8 = (a_bf16.reshape(M_tokens, K // bk, bk) / a_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M_tokens, K)
    b_nk = b_bf16.permute(0, 2, 1).contiguous()
    b_max = b_nk.abs().reshape(E, N // bn, bn, K // bk, bk).amax(dim=(2, 4)).float()
    b_scale = (b_max / 448.0).clamp(min=1e-12)
    b_fp8 = (b_nk.reshape(E, N // bn, bn, K // bk, bk) / b_scale.unsqueeze(2).unsqueeze(4)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(E, K, N)
    b_scale = b_scale.reshape(E, N // bn, K // bk).contiguous()
    return a_fp8, a_scale.to(torch.float32), b_fp8, b_scale.to(torch.float32)


def run(a_fp8, b_fp8, a_scale, b_scale, sti, eid, npp_t, nvt, npm, BM, BN, BK, NX):
    c = torch.zeros(npm * BM, N, dtype=torch.bfloat16, device=DEV)
    _launch(a_fp8, b_fp8, a_scale, b_scale, c, sti, eid, npp_t, nvt, npm, 1)
    return c


def bench(M_tokens, BM, BN, BK, NX, warmup=15, iters=60):
    a_fp8, a_scale, b_fp8, b_scale = make_blockwise_fp8(M_tokens, E, K, N, DEV, BN, BK)
    sti, eid, npm, nvt = build_routing_balanced(M_tokens, E, TOPK, DEV, BM)
    npp_t = torch.tensor([nvt], dtype=torch.int32, device=DEV)
    # round-robin reference (same kernel, full grid, no N-split)
    c_rr = torch.zeros(npm * BM, N, dtype=torch.bfloat16, device=DEV)
    _launch(a_fp8, b_fp8, a_scale, b_scale, c_rr, sti, eid, npp_t, nvt, npm, 0)
    # megakernel
    for _ in range(warmup):
        run(a_fp8, b_fp8, a_scale, b_scale, sti, eid, npp_t, nvt, npm, BM, BN, BK, NX)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        run(a_fp8, b_fp8, a_scale, b_scale, sti, eid, npp_t, nvt, npm, BM, BN, BK, NX)
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def main():
    print("=== Phase 6: decode megakernel config sweep ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K} N={N}\n")
    configs = [(16, 128, 128, 8), (32, 128, 128, 8), (64, 128, 128, 8), (128, 128, 128, 8),
              (64, 64, 128, 8), (64, 256, 128, 8), (64, 128, 64, 8), (64, 128, 256, 8),
              (64, 128, 128, 4), (64, 128, 128, 16)]
    print(f"{'BM':>4} {'BN':>4} {'BK':>4} {'NX':>4} | {'M=512':>9} {'M=1024':>10} {'M=2048':>10}")
    for (BM, BN, BK, NX) in configs:
        a_fp8, a_scale, b_fp8, b_scale = make_blockwise_fp8(512, E, K, N, DEV, BN, BK)
        sti, eid, npm, nvt = build_routing_balanced(512, E, TOPK, DEV, BM)
        npp_t = torch.tensor([nvt], dtype=torch.int32, device=DEV)
        c_mk = run(a_fp8, b_fp8, a_scale, b_scale, sti, eid, npp_t, nvt, npm, BM, BN, BK, NX)
        c_rr = torch.zeros(npm * BM, N, dtype=torch.bfloat16, device=DEV)
        _launch(a_fp8, b_fp8, a_scale, b_scale, c_rr, sti, eid, npp_t, nvt, npm, 0)
        torch.cuda.synchronize()
        valid = sti < nvt
        diff = (c_mk[valid].to(torch.float32) - c_rr[valid].to(torch.float32)).abs().max().item()
        t512 = bench(512, BM, BN, BK, NX)
        t1024 = bench(1024, BM, BN, BK, NX)
        t2048 = bench(2048, BM, BN, BK, NX)
        print(f"{BM:>4} {BN:>4} {BK:>4} {NX:>4} | {t512:>9.1f} {t1024:>10.1f} {t2048:>10.1f}  diff={diff:.4f}")


if __name__ == "__main__":
    main()
