#!/usr/bin/env python
"""Parse rocprofv2 pmc CSVs and compute per-kernel L2 hit rate + HBM traffic."""
import csv, sys, os, glob

def parse(outdir):
    # pmc_1 = L2 hit (TCC_HIT_sum, TCC_MISS_sum, SQ_WAVES)
    # pmc_2 = HBM  (FETCH_SIZE, WRITE_SIZE)
    l2_csvs = sorted(glob.glob(os.path.join(outdir, "pmc_1", "results_*.csv")))
    hbm_csvs = sorted(glob.glob(os.path.join(outdir, "pmc_2", "results_*.csv")))
    if not l2_csvs or not hbm_csvs:
        print(f"  no csvs in {outdir}"); return
    # aggregate per kernel-name-substring
    def agg(csvs, keys):
        totals = {}
        for c in csvs:
            with open(c) as f:
                r = csv.DictReader(f)
                for row in r:
                    kn = row.get("Kernel_Name","")
                    for key in keys:
                        try: v = float(row.get(key, 0) or 0)
                        except: v = 0.0
                        totals.setdefault(kn, {}).setdefault(key, 0.0)
                        totals[kn][key] += v
                    totals[kn].setdefault("count", 0)
                    totals[kn]["count"] += 1
        return totals
    l2 = agg(l2_csvs, ["TCC_HIT_sum","TCC_MISS_sum","SQ_WAVES"])
    hbm = agg(hbm_csvs, ["FETCH_SIZE","WRITE_SIZE"])
    # find fmoe + topk kernels
    print(f"=== {outdir} ===")
    for kn in l2:
        if "fmoe" in kn.lower() or "topk" in kn.lower() or "moe" in kn.lower():
            hit = l2[kn]["TCC_HIT_sum"]; miss = l2[kn]["TCC_MISS_sum"]; n = l2[kn]["count"]
            rate = 100*hit/(hit+miss) if (hit+miss)>0 else 0
            h = hbm.get(kn, {})
            fetch = h.get("FETCH_SIZE",0); write = h.get("WRITE_SIZE",0)
            short = kn.split("::")[-1].split(" (")[0].strip()
            print(f"  {short:55s} dispatches={n:4d}  L2hit={rate:6.2f}%  HIT={hit:.0f} MISS={miss:.0f}  HBMfetch={fetch:.0f}KB HBMwrite={write:.0f}KB")

for M in sys.argv[1:]:
    parse(f"/tmp/rp2_m{M}")
