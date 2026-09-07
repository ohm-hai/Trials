"""Phase 9b: full prefill MoE orchestrator (expert-sequential per-expert launch,
stage1+silu+stage2+unpermute) vs prod aiter.fused_moe hsaco, across a
(batch_size x seq_len) grid -> M = batch_size * seq_len."""
import torch, sys
sys.path.insert(0, "/scratch/aiter"); sys.path.insert(0, "/scratch/sglang/python"); sys.path.insert(0, "/root/moe_bench_results")
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from aiter.ops.triton._triton_kernels.moe.moe_prefill_megakernel import moe_prefill_stage1_expert
from aiter.fused_moe import fused_moe, QuantType, ActivationType

BM = 128; BN = 128; BK = 128


def split_by_expert(sorted_token_ids, expert_ids):
    out = []
    npm = expert_ids.shape[0]
    EM = E - 1
    eids = expert_ids.clamp(min=0, max=EM).tolist()
    e = eids[0]; start = 0
    for pm in range(1, npm):
        ee = eids[pm]
        if ee != e:
            out.append((e, start, pm))
            e = ee; start = pm
    out.append((e, start, npm))
    return out


def silu_gate_up(g1):
    gate = g1[:, 0::2]; up = g1[:, 1::2]
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def moe_prefill_full(h, w13f, w13s, w2f, w2s, tw, ti, mode=1):
    """Expert-sequential prefill orchestrator. Returns (M, K) bf16."""
    M, K = h.shape
    a1, a1s = per_token_group_quant_fp8(h, BK)
    st, ei, ntp = moe_align_block_size(ti, BM, E)
    experts = split_by_expert(st, ei)
    T = M * TOPK
    out_flat = torch.zeros(T, K, dtype=torch.float32, device=DEV)
    tw_flat = tw.to(torch.float32).reshape(-1)
    for e_id, start, end in experts:
        MB_e = end - start
        if MB_e == 0:
            continue
        sti_e = st[start*BM:end*BM].contiguous()
        # stage1: (MB_e*BM, K) @ w13_e -> (MB_e*BM, N2)
        b13_e = w13f[e_id]; bs13_e = w13s[e_id]
        g1 = torch.zeros(MB_e * BM, N2, dtype=torch.bfloat16, device=DEV)
        moe_prefill_stage1_expert(a1, b13_e, a1s, bs13_e, sti_e, g1, TOPK,
                                  block_m=BM, block_n=BN, block_k=BK, mode=mode, a_by_sorted=False)
        inter = silu_gate_up(g1)
        a2, a2s = per_token_group_quant_fp8(inter, BK)
        # stage2: (MB_e*BM, N_inter) @ w2_e -> (MB_e*BM, K)
        b2_e = w2f[e_id]; bs2_e = w2s[e_id]
        g2 = torch.zeros(MB_e * BM, K, dtype=torch.bfloat16, device=DEV)
        moe_prefill_stage1_expert(a2, b2_e, a2s, bs2_e, sti_e, g2, TOPK,
                                  block_m=BM, block_n=BN, block_k=BK, mode=mode, a_by_sorted=True)
        # unpermute: scatter this expert's rows into out_flat[token*TOPK+topk]
        st_e = sti_e  # (MB_e*BM,)
        valid = st_e < T
        vst = st_e[valid]
        vrows = g2[:st_e.shape[0]][valid]   # (num_valid_e, K)
        vw = tw_flat[vst]
        out_flat.index_add_(0, vst, vrows.to(torch.float32) * vw.unsqueeze(1))
    return out_flat.view(M, TOPK, K).sum(dim=1).to(torch.bfloat16)


def bench_fn(fn, warmup=10, iters=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def main():
    torch.manual_seed(0)
    w13_bf16 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
    w2_bf16 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
    w13f, w13s = per_block_quant_fp8(w13_bf16)
    w2f, w2s = per_block_quant_fp8(w2_bf16)
    print("=== Phase 9b: prefill megakernel (expert-sequential) vs prod hsaco ===")
    print(f"GLM-5.2 fp8 | E={E} topk={TOPK} K={K_HIDDEN} N_inter={N_INTER} BM={BM}\n")
    print(f"{'bs':>5} {'seq':>5} {'M':>7} {'mk us':>8} {'prod us':>9} {'spd':>7} {'err':>8}")
    # (batch_size, seq_len) grid -> M = bs*seq, covering prefill chunks + concurrency
    grid = [(1,2048),(1,4096),(1,8192),(2,2048),(4,1024),(4,2048),
             (8,512),(8,1024),(8,2048),(16,256),(16,512),(16,1024),
             (32,128),(32,256),(32,512),(64,64),(64,128),(64,256)]
    for bs, seq in grid:
        M = bs * seq
        h = torch.randn(M, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
        ti = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9
        # correctness vs prod
        out_mk = moe_prefill_full(h, w13f, w13s, w2f, w2s, tw, ti, mode=1)
        def prod():
            return fused_moe(hidden_states=h, w1=w13f, w2=w2f, topk_weight=tw.to(torch.float32),
                               topk_ids=ti, quant_type=QuantType.per_128x128,
                               activation=ActivationType.Silu,
                               w1_scale=w13s, w2_scale=w2s, a1_scale=None, a2_scale=None)
        try:
            out_p = prod()
            err = (out_mk.to(torch.float32) - out_p.to(torch.float32)).abs().max().item()
            t_mk = bench_fn(lambda: moe_prefill_full(h, w13f, w13s, w2f, w2s, tw, ti, mode=1))
            t_p = bench_fn(prod)
            print(f"{bs:>5} {seq:>5} {M:>7} {t_mk:>8.1f} {t_p:>9.1f} {t_p/t_mk:>6.2f}x {err:>8.3f}")
        except Exception as ex:
            print(f"{bs:>5} {seq:>5} {M:>7} {'ERR':>8} {str(ex)[:30]:>9}")


if __name__ == "__main__":
    main()
