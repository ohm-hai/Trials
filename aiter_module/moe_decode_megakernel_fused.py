"""EP-between-XCDs decode megakernel -- FUSED variant for tiny-M decode.

Two epilogue fusions over the base megakernel:
  * FUSED_SILU (stage-1): writes inter = silu(gate)*up (N_inter cols) directly
    from the interleaved gate-up accumulator. Eliminates 3 separate silu/mul/cast
    launches and the gemm1 intermediate.
  * FUSED_UNPERMUTE (stage-2): applies router weight and atomic-adds each row
    into the final (M, K) output at the original token index. Eliminates ~8
    unpermute/weight/scatter/sum launches and the gemm2 intermediate.

Collapses the orchestrator from ~15 launches to 2, removing the ~240us of launch
overhead that made the unfused orchestrator regress at tiny decode batches.
"""
import torch
import triton
import triton.language as tl

NUM_XCD = 8


@triton.jit
def _moe_decode_mega_kernel_fused(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr,
    c_ptr, sti_ptr, eid_ptr, nvt,
    N, K, EM,
    sam, sak, sbe, sbk, sbn, scm, scn,
    sasm, sask, sbse, sbsn, sbsk,
    out_ptr, tw_ptr, som, sok,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, NX: tl.constexpr, TOPK: tl.constexpr,
    NBPS: tl.constexpr, NPM: tl.constexpr, GN: tl.constexpr, GK: tl.constexpr,
    FUSED_SILU: tl.constexpr, FUSED_UNPERMUTE: tl.constexpr,
):
    pid = tl.program_id(0)
    xcd = pid // (NPM * NBPS)
    local = pid % (NPM * NBPS)
    pid_m = local // NBPS
    nb_in_slice = local % NBPS
    pid_n = xcd * NBPS + nb_in_slice
    if pid_m >= NPM:
        return
    oti = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    ot = tl.load(sti_ptr + oti)
    tm = ot < nvt
    ot = tl.where(tm, ot, 0)
    oe = tl.load(eid_ptr + pid_m.to(tl.int64)).to(tl.int64)
    oe = tl.minimum(oe, EM - 1)
    obn = (pid_n * BN + tl.arange(0, BN).to(tl.int64)) % N
    ok = tl.arange(0, BK)
    # Stage-1: a = hidden (M rows, original-token order) -> index by ot//TOPK.
    # Stage-2 (FUSED_UNPERMUTE): a = inter (npm*BM rows, SORTED order) -> index
    # by sorted position oti. a_scale follows the same row indexing.
    if FUSED_UNPERMUTE:
        a_row = oti                       # sorted position
        a_scale_row = oti
    else:
        a_row = ot // TOPK                 # original token
        a_scale_row = ot // TOPK
    ap = a_ptr + (a_row[:, None] * sam + ok[None, :] * sak)
    bp = b_ptr + (oe * sbe + ok[:, None] * sbk + obn[None, :] * sbn)
    offs_bsn = obn // GN
    a_scale_ptrs = a_scale_ptr + a_scale_row * sasm
    b_scale_ptrs = b_scale_ptr + oe * sbse + offs_bsn * sbsn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a_fp8 = tl.load(ap, mask=tm[:, None])
        b_fp8 = tl.load(bp)
        offs_ks = (kk * BK) // GK
        a_scale = tl.load(a_scale_ptrs + offs_ks * sask, mask=tm, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * sbsk)
        acc += tl.dot(a_fp8, b_fp8) * a_scale[:, None] * b_scale[None, :]
        ap += BK * sak
        bp += BK * sbk
    c = acc.to(tl.bfloat16)

    if FUSED_SILU:
        # gate-up INTERLEAVED: cols 0,2,..=gate; 1,3,..=up. inter=silu(gate)*up.
        # Reshape (BM, BN) -> (BM, BN//2, 2) to deinterleave gate/up pairs.
        cpair = tl.reshape(c.to(tl.float32), (BM, BN // 2, 2))
        gate = tl.sum(tl.where(tl.arange(0, 2)[None, None, :] == 0, cpair, 0.0), axis=2)
        up = tl.sum(tl.where(tl.arange(0, 2)[None, None, :] == 1, cpair, 0.0), axis=2)
        inter = (tl.sigmoid(gate) * gate * up).to(tl.bfloat16)
        ocm = pid_m * BM + tl.arange(0, BM).to(tl.int64)
        ocn = pid_n * (BN // 2) + tl.arange(0, BN // 2).to(tl.int64)
        cp = c_ptr + ocm[:, None] * scm + ocn[None, :] * scn
        tl.store(cp, inter, mask=tm[:, None])
    elif FUSED_UNPERMUTE:
        # router weight + atomic-add into final (M, K) output at token index.
        tok = ot // TOPK
        w = tl.load(tw_ptr + ot.to(tl.int64))
        ocn = pid_n * BN + tl.arange(0, BN).to(tl.int64)
        op = out_ptr + tok[:, None] * som + ocn[None, :] * sok
        val = c.to(tl.float32) * w[:, None]
        tl.atomic_add(op, val, mask=tm[:, None])
    else:
        ocm = pid_m * BM + tl.arange(0, BM).to(tl.int64)
        ocn = pid_n * BN + tl.arange(0, BN).to(tl.int64)
        cp = c_ptr + ocm[:, None] * scm + ocn[None, :] * scn
        tl.store(cp, c, mask=tm[:, None])


def _launch(a_fp8, b_fp8, a_scale, b_scale, c_ptr,
            sti, eid, nvt, N, K, E, scm, scn,
            out_ptr, tw_ptr, som, sok,
            BM, BN, BK, TOPK, fused_silu, fused_unpermute):
    npm = eid.shape[0]
    n_slice = N // NUM_XCD
    nbps = n_slice // BN
    grid = (NUM_XCD * npm * nbps,)
    _moe_decode_mega_kernel_fused[grid](
        a_fp8, b_fp8, a_scale, b_scale, c_ptr,
        sti, eid, nvt, N, K, E,
        a_fp8.stride(0), a_fp8.stride(1),
        b_fp8.stride(0), b_fp8.stride(2), b_fp8.stride(1),
        scm, scn,
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1), b_scale.stride(2),
        out_ptr, tw_ptr, som, sok,
        BM, BN, BK, 8, NUM_XCD, TOPK, nbps, npm, 128, 128,
        fused_silu, fused_unpermute,
    )


def moe_decode_megakernel_stage1_fused(
    a_fp8, b_fp8, a_scale, b_scale,
    sorted_token_ids, expert_ids, out, topk,
    block_m=64, block_n=128, block_k=128, fused_silu=False,
):
    """Stage-1 GEMM. If fused_silu, `out` is (npm*BM, N_inter) and the kernel
    writes silu(gate)*up directly (b_fp8 is still (E, N2, K))."""
    M, K = a_fp8.shape
    E, N, _ = b_fp8.shape
    nvt = M * topk
    _launch(a_fp8, b_fp8, a_scale, b_scale, out,
            sorted_token_ids, expert_ids, nvt, N, K, E,
            out.stride(0), out.stride(1),
            out, out, 0, 0,  # unused for silu/plain stage
            block_m, block_n, block_k, topk, fused_silu, False)


def moe_decode_megakernel_stage2_fused(
    a_fp8, b_fp8, a_scale, b_scale,
    sorted_token_ids, expert_ids, out_final, topk_weights, topk,
    block_m=64, block_n=128, block_k=128,
):
    """Stage-2 GEMM with fused unpermute. `out_final` is (M, K) fp32, zero-init.
    Each block atomic-adds its weighted result into out_final[token, :]."""
    M, K = a_fp8.shape
    E, N, _ = b_fp8.shape
    nvt = M * topk
    _launch(a_fp8, b_fp8, a_scale, b_scale, out_final,
            sorted_token_ids, expert_ids, nvt, N, K, E,
            out_final.stride(0), out_final.stride(1),
            out_final, topk_weights.reshape(-1),
            out_final.stride(0), out_final.stride(1),
            block_m, block_n, block_k, topk, False, True)

