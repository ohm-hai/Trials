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

torch.manual_seed(0)
BS = 64
w13 = torch.randn(E, N2, K_HIDDEN, device=DEV, dtype=torch.bfloat16) * 0.02
w13f, w13s = per_block_quant_fp8(w13)
h = torch.randn(BS, K_HIDDEN, device=DEV, dtype=torch.bfloat16) * 0.1
ti = torch.randint(0, E, (BS, TOPK), dtype=torch.int32, device=DEV)

BM = 128; BN = 128; BK = 128
sti, eid, ntp_t = moe_align_block_size(ti, BM, E)
ntp = ntp_t.item()
npm_cap = eid.shape[0]
npm_valid = ntp // BM
print(f"ntp={ntp} npm_cap={npm_cap} npm_valid={npm_valid} sti.len={sti.shape[0]}", flush=True)
print(f"eid[:5]={eid[:5].tolist()} eid[-5:]={eid[-5:].tolist()} (tail garbage)", flush=True)
print(f"eid.min={eid.min().item()} eid.max={eid.max().item()}", flush=True)

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
a1_fp8, a1_scale = per_token_group_quant_fp8(h, BK)

def launch(npm, tag):
    gemm1 = torch.zeros(npm * BM, N2, device=DEV, dtype=torch.bfloat16)
    t = time.time()
    moe_decode_megakernel_stage1(
        a1_fp8, w13f, a1_scale, w13s, sti, eid, gemm1, TOPK,
        block_m=BM, block_n=BN, block_k=BK,
    )
    torch.cuda.synchronize()
    print(f"[{tag}] npm={npm} grid={8*npm*(N2//BN)} -> {round(time.time()-t,3)}s max={gemm1.abs().max().item():.4f}", flush=True)

# Valid npm (should work)
launch(npm_valid, "valid-npm=222")
# Capacity npm (may hang due to garbage eid tail)
print("launching capacity npm=259 (may hang)...", flush=True)
launch(npm_cap, "cap-npm=259")
print("DONE", flush=True)
