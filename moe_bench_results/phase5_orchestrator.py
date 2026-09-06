"""Phase 5: full decode MoE orchestrator using the EP-between-XCDs megakernel for BOTH
stage1 (gate-up) and stage2 (down-proj), PRODUCTION (E, N, K) fp8 layout, block-wise 128x128.
GLM-5.2: E=256, topk=8, K_hidden=6144, N_inter=2048.
  w13: (E, 2*N_inter, K_hidden) fp8  -- stage1 gate-up
  w2:  (E, K_hidden, N_inter) fp8    -- stage2 down-proj
Both are (E, N, K) in the megakernel convention (N=out-width, K=in-width)."""
import torch
import sys
sys.path.insert(0, "/scratch/aiter")
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel import moe_decode_megakernel_stage1

DEV = "cuda"
NUM_XCD = 8
E = 256
TOPK = 8
K_HIDDEN = 6144
N_INTER = 2048
N2 = 2 * N_INTER
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
GROUP_N = 128
GROUP_K = 128


def per_token_group_quant_fp8(x, gk=128):
    M, K = x.shape
    xg = x.reshape(M, K // gk, gk)
    amax = xg.abs().amax(dim=-1).float()
    scale = (amax / 448.0).clamp(min=1e-12)
    q = (xg / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(M, K)
    return q, scale


def per_block_quant_fp8(w, bn=128, bk=128):
    E, N, K = w.shape
    wg = w.reshape(E, N // bn, bn, K // bk, bk)
    amax = wg.abs().amax(dim=(2, 4)).float()
    scale = (amax / 448.0).clamp(min=1e-12)
    q = (wg / scale.unsqueeze(2).unsqueeze(4)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(E, N, K)
    return q, scale


def silu_gate_up(g1, N_inter):
    # production w13 is gate-up INTERLEAVED (gate_up_interleaved=True): gate/up alternate columns
    gate = g1[:, 0::2]
    up = g1[:, 1::2]
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def build_routing_from_topk(topk_ids, device):
    M, topk = topk_ids.shape
    flat_idx = topk_ids.reshape(-1).cpu()
    flat_token = torch.arange(M * topk, dtype=torch.int32)
    order = torch.argsort(flat_idx, stable=True)
    sorted_expert = flat_idx[order].to(torch.int32)
    sorted_token = flat_token[order].to(torch.int32).to(device)
    npm = (M * topk + BLOCK_M - 1) // BLOCK_M
    eid = torch.empty(npm, dtype=torch.int32, device=device)
    for pm in range(npm):
        pos = pm * BLOCK_M
        eid[pm] = sorted_expert[pos] if pos < len(sorted_expert) else 0
    pad = npm * BLOCK_M - len(sorted_token)
    if pad > 0:
        sorted_token = torch.cat([sorted_token, torch.full((pad,), M * topk, dtype=torch.int32, device=device)])
    return sorted_token, eid, npm


def moe_decode_full(hidden, w13_fp8, w13_scale, w2_fp8, w2_scale,
                     topk_weights, sorted_token, eid, npm):
    M = hidden.shape[0]
    T = M * TOPK
    a1, a1s = per_token_group_quant_fp8(hidden)
    g1 = torch.empty(npm * BLOCK_M, N2, dtype=torch.bfloat16, device=DEV)
    moe_decode_megakernel_stage1(a1, w13_fp8, a1s, w13_scale, sorted_token, eid, g1, TOPK)
    inter = silu_gate_up(g1, N_INTER)
    a2, a2s = per_token_group_quant_fp8(inter)
    g2 = torch.empty(npm * BLOCK_M, K_HIDDEN, dtype=torch.bfloat16, device=DEV)
    moe_decode_megakernel_stage1(a2, w2_fp8, a2s, w2_scale, sorted_token, eid, g2, TOPK)
    st = sorted_token.clamp(max=T - 1)
    acc = g2[st] * topk_weights.reshape(-1, 1).to(torch.bfloat16)
    return acc.view(M, TOPK, K_HIDDEN).sum(dim=1)


def bf16_ref(hidden, w13_bf16, w2_bf16, topk_ids, topk_weights):
    M, K = hidden.shape
    out = torch.zeros(M, K, dtype=torch.bfloat16, device=DEV)
    for t in range(M):
        for k in range(TOPK):
            e = int(topk_ids[t, k].item())
            h = hidden[t].to(torch.float32)
            g = w13_bf16[e, 0::2].to(torch.float32)
            u = w13_bf16[e, 1::2].to(torch.float32)
            inter = (torch.nn.functional.silu(h @ g.T) * (h @ u.T))
            o = inter @ w2_bf16[e].to(torch.float32).T
            out[t] += o.to(torch.bfloat16) * topk_weights[t, k].to(torch.bfloat16)
    return out


def main():
    torch.manual_seed(0)
    M = 512
    hidden = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
    w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
    w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
    topk_weights = torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
    sorted_token, eid, npm = build_routing_from_topk(topk_ids, DEV)
    w13_fp8, w13_scale = per_block_quant_fp8(w13_bf16)
    w2_fp8, w2_scale = per_block_quant_fp8(w2_bf16)
    out_mk = moe_decode_full(hidden, w13_fp8, w13_scale, w2_fp8, w2_scale,
                             topk_weights, sorted_token, eid, npm)
    out_ref = bf16_ref(hidden, w13_bf16, w2_bf16, topk_ids, topk_weights)
    torch.cuda.synchronize()
    err = (out_mk.to(torch.float32) - out_ref.to(torch.float32)).abs().max().item()
    rel = err / (out_ref.to(torch.float32).abs().mean().item() + 1e-9)
    print(f"full decode MoE @ M={M}: max_abs_err={err:.4f} rel={rel:.4f} (fp8-megakernel vs bf16 ref)")
    print(f"\n{'batch':>7} {'npm':>5} {'megakernel-full us':>20}")
    for bs in [256, 512, 1024, 2048, 4096, 8192]:
        h = torch.randn(bs, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (bs, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(bs, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        st, ei, np_ = build_routing_from_topk(ti, DEV)
        w13f, w13s = per_block_quant_fp8(torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02)
        w2f, w2s = per_block_quant_fp8(torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02)
        for _ in range(15):
            moe_decode_full(h, w13f, w13s, w2f, w2s, tw, st, ei, np_)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(40):
            moe_decode_full(h, w13f, w13s, w2f, w2s, tw, st, ei, np_)
        t1.record(); torch.cuda.synchronize()
        print(f"{bs:>7} {np_:>5} {t0.elapsed_time(t1)*1e3/40:>20.1f}")


if __name__ == "__main__":
    main()
