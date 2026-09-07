import torch, sys, time
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
w13 = torch.randn(E, N2, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.02
w2 = torch.randn(E, K_HIDDEN, N_INTER, dtype=torch.bfloat16, device=DEV) * 0.02
w13f, w13s = per_block_quant_fp8(w13)
w2f, w2s = per_block_quant_fp8(w2)
h = torch.randn(BS, K_HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.1
ti = torch.randint(0, E, (BS, TOPK), dtype=torch.int32, device=DEV)
tw = torch.rand(BS, TOPK, dtype=torch.bfloat16, device=DEV) * 0.1 + 0.9

BM = 128
BK = 128
BN = 128

def tlog(msg):
    print(f"[{time.time()-T0:7.2f}] {msg}", flush=True)

T0 = time.time()
tlog("start")

# Step 1: moe_align_block_size
sorted_token_ids, expert_ids, ntp = moe_align_block_size(ti, BM, E)
torch.cuda.synchronize()
tlog(f"moe_align done: npm={expert_ids.shape[0]} ntp={ntp}")

# Step 2: per_token_group_quant_fp8 stage1
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
a1_fp8, a1_scale = per_token_group_quant_fp8(h, BK)
torch.cuda.synchronize()
tlog(f"stage1 quant done: a1_fp8={a1_fp8.shape} {a1_fp8.dtype}")

# Step 3: stage1 GEMM
npm = expert_ids.shape[0]
gemm1 = torch.zeros(npm * BM, N2, dtype=torch.bfloat16, device=DEV)
tlog(f"stage1 GEMM launch: grid will be npm={npm} * (N2/BN)={N2//BN} = {npm * (N2//BN)}")
moe_decode_megakernel_stage1(
    a1_fp8, w13f, a1_scale, w13s,
    sorted_token_ids, expert_ids, gemm1, TOPK,
    block_m=BM, block_n=BN, block_k=BK,
)
torch.cuda.synchronize()
tlog(f"stage1 GEMM done: gemm1 max={gemm1.abs().max().item():.4f}")

# Step 4: SiLU
gate = gemm1[:, 0::2]
up = gemm1[:, 1::2]
inter = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
torch.cuda.synchronize()
tlog("silu done")

# Step 5: stage2 quant
a2_fp8, a2_scale = per_token_group_quant_fp8(inter, BK)
torch.cuda.synchronize()
tlog(f"stage2 quant done: a2_fp8={a2_fp8.shape}")

# Step 6: stage2 GEMM (a_by_sorted=True) -- THIS is the suspected hang
gemm2 = torch.zeros(npm * BM, K_HIDDEN, dtype=torch.bfloat16, device=DEV)
tlog(f"stage2 GEMM launch (a_by_sorted=True): grid={npm * (K_HIDDEN//BN)}")
moe_decode_megakernel_stage1(
    a2_fp8, w2f, a2_scale, w2s,
    sorted_token_ids, expert_ids, gemm2, TOPK,
    block_m=BM, block_n=BN, block_k=BK,
    a_by_sorted=True,
)
torch.cuda.synchronize()
tlog(f"stage2 GEMM done: gemm2 max={gemm2.abs().max().item():.4f}")

# Step 7: unpermute
T = BS * TOPK
st = sorted_token_ids
valid = st < T
pos = torch.arange(st.shape[0], device=DEV)
vpos = pos[valid]
vst = st[valid]
vrows = gemm2[vpos]
vw = tw.reshape(-1)[vst].to(torch.bfloat16)
weighted = vrows * vw.unsqueeze(1)
out_flat = torch.zeros(T, K_HIDDEN, device=DEV, dtype=torch.float32)
out_flat.index_add_(0, vst, weighted.to(torch.float32))
out = out_flat.view(BS, TOPK, K_HIDDEN).sum(dim=1).to(torch.bfloat16)
torch.cuda.synchronize()
tlog(f"unpermute done: out max={out.abs().max().item():.4f}")
print("DONE", flush=True)
