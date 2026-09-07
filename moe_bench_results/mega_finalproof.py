import torch, time, sys
sys.path.insert(0, '/scratch/aiter')
sys.path.insert(0, '/scratch/sglang/python')
sys.path.insert(0, '/root/moe_bench_results')
from phase5_orchestrator import per_block_quant_fp8, E, TOPK, K_HIDDEN, N_INTER, N2, DEV
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from aiter.ops.triton._triton_kernels.moe.moe_decode_megakernel import (
    moe_decode_megakernel_stage1,
)
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

# Reproduce stepdiag's EXACT allocation order (includes w2) so garbage matches
torch.manual_seed(0)
BS = 64
w13 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
w2 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
w13f, w13s = per_block_quant_fp8(w13)
w2f, w2s = per_block_quant_fp8(w2)
h = torch.randn(BS, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
ti = torch.randint(0, E, (BS, TOPK), dtype=torch.int32, device=DEV)
tw = torch.rand(BS, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9

BM = 128; BN = 128; BK = 128
sti, eid, ntp_t = moe_align_block_size(ti, BM, E)
ntp = ntp_t.item()
npm_cap = eid.shape[0]
npm_valid = ntp // BM
print(f"ntp={ntp} npm_cap={npm_cap} npm_valid={npm_valid}", flush=True)
print(f"eid.min={eid.min().item()} eid.max={eid.max().item()}", flush=True)
print(f"eid tail (last 10)={eid[-10:].tolist()}", flush=True)

a1_fp8, a1_scale = per_token_group_quant_fp8(h, BK)

def launch(npm, eid_arg, sti_arg, tag, a_by_sorted=False):
    gemm1 = torch.zeros(npm * BM, N2, device=DEV, dtype=torch.bfloat16)
    t = time.time()
    moe_decode_megakernel_stage1(
        a1_fp8, w13f, a1_scale, w13s, sti_arg, eid_arg, gemm1, TOPK,
        block_m=BM, block_n=BN, block_k=BK, a_by_sorted=a_by_sorted,
    )
    torch.cuda.synchronize()
    print(f"[{tag}] npm={npm} -> {round(time.time()-t,3)}s max={gemm1.abs().max().item():.4f}", flush=True)

# (a) raw capacity npm=259 -- SKIPPED: proven to hang (negative garbage eid -> OOB)
# print("TEST(a): raw cap npm=259", flush=True)
# launch(npm_cap, eid, sti, "raw-cap259")

# (b) valid npm = ntp//BM -- should WORK (no garbage read)
print("TEST(b): valid npm", flush=True)
launch(npm_valid, eid, sti, "valid-npm")

# (c) clamped eid + sentinel sti tail, cap npm -- should WORK
eid_c = eid.clamp(0, E - 1).contiguous()
sti_s = sti.clone(); sti_s[ntp:] = BS * TOPK
print("TEST(c): clamped cap npm=259", flush=True)
launch(npm_cap, eid_c, sti_s, "clamped-cap259")

print("DONE", flush=True)
