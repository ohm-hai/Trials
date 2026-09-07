#!/usr/bin/env python
"""Measure GLM-5.2 MoE routing skew and XCD imbalance under expert-affine (e%8).

Loads the real gate.weight + e_score_correction_bias for several sparse MoE
layers, runs the exact GLM-5.2 routing (sigmoid + correction_bias, global top-8,
n_group=1/topk_group=1 per topk.py:1873-1884) on a large batch of Gaussian
hidden states (post-LayerNorm approximation), and computes:
  - per-expert frequency (tokens routed to each expert)
  - per-XCD load under expert-affine mapping xcd = e % 8
  - imbalance ratio (max/min) for experts and for XCDs
  - the L2-reuse implication (tokens/expert, tokens/XCD)

Caveat: inputs are Gaussian N(0,1) (post-LayerNorm approximation), not real
activations. The gate *structure* (which experts are hot) is real; the absolute
frequencies are approximate. The correction_bias is the real learned bias.
"""
import json, os, sys, glob
import torch
from safetensors import safe_open

D = "/scratch/hf/hub/models--zai-org--GLM-5.2-FP8/snapshots/f33c6dc501ee5a2c7e35155653b1b1abbc320951"
idx = json.load(open(f"{D}/model.safetensors.index.json"))
wm = idx["weight_map"]

NUM_XCD = 8
E = 256
TOPK = 8
LAYERS = [10, 20, 30, 40, 50, 60]   # sparse MoE layers (first_k_dense_replace=3, freq=1 -> layer>=3? but experts start at 10 per index)
BATCH = 16384                         # tokens (large for stable frequencies)
HID = 6144

def load_gate(layer):
    gkey = f"model.layers.{layer}.mlp.gate.weight"
    bkey = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    shard = wm[gkey]
    path = f"{D}/{shard}"
    w = b = None
    with safe_open(path, framework="pt", device="cpu") as f:
        if gkey in f.keys():
            w = f.get_tensor(gkey).to(torch.float32)
        if bkey in f.keys():
            b = f.get_tensor(bkey).to(torch.float32)
    # bias may be in a different shard
    if b is None:
        bshard = wm[bkey]
        with safe_open(f"{D}/{bshard}", framework="pt", device="cpu") as f:
            b = f.get_tensor(bkey).to(torch.float32)
    return w, b

def route(w, b, x):
    # scores = sigmoid(x @ W.T) + correction_bias ; global top-8
    logits = x @ w.t()                  # (B, E)
    scores = torch.sigmoid(logits) + b  # (B, E)
    topk_ids = torch.topk(scores, k=TOPK, dim=-1, sorted=False).indices  # (B, 8)
    return topk_ids

def analyze(layer, w, b, x):
    topk_ids = route(w, b, x)                       # (B, 8)
    flat = topk_ids.reshape(-1)                     # (B*8,)
    exp_cnt = torch.bincount(flat, minlength=E).to(torch.float64)  # (E,)
    # per-expert
    emean = exp_cnt.mean().item(); emax = exp_cnt.max().item(); emin = exp_cnt.min().item()
    estd = exp_cnt.std().item()
    # per-XCD under e%8
    xcd_cnt = exp_cnt.view(NUM_XCD, E // NUM_XCD).sum(dim=1)   # (8,) sum of 32 experts each
    xmean = xcd_cnt.mean().item(); xmax = xcd_cnt.max().item(); xmin = xcd_cnt.min().item()
    xstd = xcd_cnt.std().item()
    # imbalance
    e_imb = emax / max(emin, 1)
    x_imb = xmax / max(xmin, 1)
    # tokens/expert and tokens/XCD (per forward, per layer)
    tpe = (BATCH * TOPK) / E
    tpx = (BATCH * TOPK) / NUM_XCD
    # top-5 hottest experts
    top5 = torch.topk(exp_cnt, 5).indices.tolist()
    bot5 = torch.topk(exp_cnt, 5, largest=False).indices.tolist()
    print(f"--- layer {layer} ---")
    print(f"  gate.weight {tuple(w.shape)} bias {tuple(b.shape)}  bias range [{b.min():.3f},{b.max():.3f}]")
    print(f"  per-expert freq: mean={emean:.1f} min={emin:.0f} max={emax:.0f} std={estd:.1f}  imbalance={e_imb:.2f}x")
    print(f"  per-XCD (e%8) load: mean={xmean:.0f} min={xmin:.0f} max={xmax:.0f} std={xstd:.1f}  imbalance={x_imb:.2f}x")
    print(f"  tokens/expert={tpe:.1f}  tokens/XCD={tpx:.1f}  (B={BATCH}, topk={TOPK})")
    print(f"  hottest experts: {top5}  coldest: {bot5}")
    # which XCDs are hot
    xcd_hot = torch.topk(xcd_cnt, 3).indices.tolist()
    xcd_cold = torch.topk(xcd_cnt, 3, largest=False).indices.tolist()
    print(f"  hot XCDs: {xcd_hot}  cold XCDs: {xcd_cold}")
    return xcd_cnt, exp_cnt

def main():
    torch.manual_seed(0)
    # post-LayerNorm-ish inputs: N(0,1). Also test a second seed for stability.
    x = torch.randn(BATCH, HID, dtype=torch.float32)
    print(f"=== GLM-5.2 routing skew & XCD imbalance (expert-affine e%8) ===")
    print(f"E={E} topk={TOPK} NUM_XCD={NUM_XCD} BATCH={BATCH} HID={HID}  inputs=N(0,1) seed0")
    all_xcd = []
    for L in LAYERS:
        w, b = load_gate(L)
        if w is None:
            print(f"layer {L}: gate weight not found, skip"); continue
        xcd_cnt, _ = analyze(L, w, b, x)
        all_xcd.append(xcd_cnt)
    # aggregate across layers (mean per-XCD load)
    if all_xcd:
        agg = torch.stack(all_xcd).mean(dim=0)
        print(f"\n=== aggregated across {len(all_xcd)} layers (mean per-XCD load) ===")
        print(f"  per-XCD mean load: {[f'{v:.0f}' for v in agg.tolist()]}")
        print(f"  imbalance={agg.max().item()/max(agg.min().item(),1):.2f}x  std={agg.std().item():.1f}")
        # coefficient of variation
        cv = agg.std().item() / agg.mean().item() * 100
        print(f"  CV={cv:.2f}%  (0% = perfectly balanced)")

if __name__ == "__main__":
    main()
