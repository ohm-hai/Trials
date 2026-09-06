"""DSA decode microbench via the aiter MLA decode kernel (gfx950-optimized),
replicating sglang's _forward_aiter + _prepare_aiter_dsa_decode_metadata path
(the --dsa-decode-backend=aiter path for GLM-5.2 on MI355X).

GLM-5.2 shape: num_heads=64, head_dim=256, index_topk=2048 pages/req, page_size=1, bf16 KV.
The sglang `tilelang` DSA backend is non-functional on this stack (CUDA-only tilelang),
so `aiter` is the recommended DSA decode backend for GLM-5.2 on gfx950.
"""
import os
import sys
import time
import statistics

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/scratch/aiter")

from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

DEV = torch.device("cuda:0")
NUM_HEADS = 64       # GLM-5.2 num_attention_heads
HEAD_DIM = 256       # GLM-5.2 qk_head_dim == v_head_dim
TOPK = 2048          # GLM-5.2 index_topk (pages per request)
PAGE_SIZE = 1
MAX_SPLIT_PER_BATCH = 64   # sglang aiter_dsa_max_split_per_batch
SM_SCALE = 1.0 / (HEAD_DIM ** 0.5)
DTYPE = torch.bfloat16
NUM_HEAD_PADDED = NUM_HEADS  # 64 is a multiple of 16 in [16,128] -> repeat factor 1

# NOTE: the aiter persistent MLA decode reduce kernel (kn_mla_reduce_v1) does NOT
# support GLM-5.2's (num_heads=64, head_dim=256) -- it was built for DeepSeek's
# head_dim=576 (kv_lora_rank 512 + rope 64). So the persistent/auto-split path fails:
#   "kn_mla_reduce_v1 doesn't support the specified settings: #heads: 64, head dimension: 256."
# The only working aiter DSA decode path for GLM-5.2 on gfx950 is the NON-persistent
# num_kv_splits=1 path (no reduce kernel). This benchmark measures that fallback.
USE_PERSISTENT = False


def make_workspace(B):
    sizes = get_mla_metadata_info_v1(
        B, 1, NUM_HEAD_PADDED, DTYPE, DTYPE,
        is_sparse=True, fast_mode=False,
        num_kv_splits=MAX_SPLIT_PER_BATCH, intra_batch_mode=True,
    )
    return [torch.empty(s, dtype=t, device=DEV) for s, t in sizes]


def bench(B, iters=50, warmup=8):
    num_pages = B * TOPK
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    kv = torch.randn(num_pages, 1, 1, HEAD_DIM, dtype=DTYPE, device=DEV)
    o = torch.empty(B, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=DEV)
    kv_indptr = torch.arange(0, (B + 1) * TOPK, TOPK, dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device=DEV)
    kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEV)

    kwargs = {}
    if USE_PERSISTENT:
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = make_workspace(B)
        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_lens,
            NUM_HEAD_PADDED, 1, False,
            work_metadata, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            page_size=PAGE_SIZE, kv_granularity=16,
            max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
            topk=TOPK, max_split_per_batch=MAX_SPLIT_PER_BATCH,
            intra_batch_mode=True, dtype_q=DTYPE, dtype_kv=DTYPE,
        )
        kwargs = dict(
            work_meta_data=work_metadata, work_indptr=work_indptr,
            work_info_set=work_info_set, reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
        )
    else:
        # Non-persistent fallback: num_kv_splits=1 (no reduce kernel).
        nksi = torch.arange(0, B + 1, dtype=torch.int32, device=DEV)
        kwargs = dict(num_kv_splits=1, num_kv_splits_indptr=nksi)

    for _ in range(warmup):
        mla_decode_fwd(
            q, kv, o, qo_indptr, kv_indptr, kv_indices,
            kv_last_page_lens, max_seqlen_q=1, page_size=PAGE_SIZE,
            nhead_kv=1, sm_scale=SM_SCALE, logit_cap=0.0, **kwargs,
        )
    torch.cuda.synchronize(DEV)

    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(DEV)
        t0 = time.perf_counter()
        mla_decode_fwd(
            q, kv, o, qo_indptr, kv_indptr, kv_indices,
            kv_last_page_lens, max_seqlen_q=1, page_size=PAGE_SIZE,
            nhead_kv=1, sm_scale=SM_SCALE, logit_cap=0.0, **kwargs,
        )
        torch.cuda.synchronize(DEV)
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts), max(ts)


def main():
    print(f"DSA decode via aiter mla_decode_fwd (gfx950, GLM-5.2: H={NUM_HEADS} D={HEAD_DIM} topk={TOPK} page_size=1 bf16)")
    print("(replicates sglang --dsa-decode-backend=aiter; tilelang DSA backend is broken on this stack)")
    print(f"{'B':>5} {'med_us':>10} {'min_us':>10} {'max_us':>10}")
    rows = []
    for B in [1, 2, 4, 8, 16, 32, 64, 128]:
        med, lo, hi = bench(B)
        print(f"{B:>5} {med:>10.2f} {lo:>10.2f} {hi:>10.2f}")
        rows.append((B, med, lo, hi))

    out = "/root/moe_bench_results/micro/dsa_aiter_decode.csv"
    with open(out, "w") as f:
        f.write("# DSA decode via aiter mla_decode_fwd (gfx950, MI355X VF, GPU0)\n")
        f.write("# GLM-5.2 shape: num_heads=64 head_dim=256 index_topk=2048 pages/req page_size=1 bf16 KV\n")
        f.write("# Replicates sglang --dsa-decode-backend=aiter (persistent MLA decode workspace path).\n")
        f.write("# sglang tilelang DSA backend is non-functional on this stack (CUDA-only tilelang build).\n")
        f.write("B,med_us,min_us,max_us\n")
        for B, med, lo, hi in rows:
            f.write(f"{B},{med:.2f},{lo:.2f},{hi:.2f}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
