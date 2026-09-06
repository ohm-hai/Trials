#!/usr/bin/env python
"""Parse rocprofv2 pmc CSVs from /tmp/rp2s_m{M} (scale sweep) -> L2 hit + HBM."""
import csv, sys, os, glob

def parse(outdir):
    l2_csvs = sorted(glob.glob(os.path.join(outdir, "pmc_1", "results_*.csv")))
    hbm_csvs = sorted(glob.glob(os.path.join(outdir, "pmc_2", "results_*.csv")))
    print(f"=== {outdir} ===  l2={len(l2_csvs)} hbm={len(hbm_csvs)}")
    if not l2_csvs or not hbm_csvs:
        print("  no csvs"); return
    keys_l2 = ["TCC_HIT_sum", "TCC_MISS_sum", "SQ_WAVES"]
    keys_hbm = ["FETCH_SIZE", "WRITE_SIZE"]
    def agg(csvs, keys):
        t = {}
        for c in csvs:
            with open(c) as f:
                for row in csv.DictReader(f):
                    kn = row.get("Kernel_Name", "")
                    for key in keys:
                        try: v = float(row.get(key, 0) or 0)
                        except: v = 0.0
                        t.setdefault(kn, {}).setdefault(key, 0.0)
                        t[kn][key] += v
                    t.setdefault(kn, {}).setdefault("count", 0)
                    t[kn]["count"] += 1
        return t
    l2 = agg(l2_csvs, keys_l2)
    hbm = agg(hbm_csvs, keys_hbm)
    for kn in l2:
        if "fmoe" in kn.lower() or "moe" in kn.lower() or "topk" in kn.lower():
            hit = l2[kn]["TCC_HIT_sum"]; miss = l2[kn]["TCC_MISS_sum"]; n = l2[kn]["count"]
            rate = 100*hit/(hit+miss) if (hit+miss)>0 else 0
            h = hbm.get(kn, {})
            fetch = h.get("FETCH_SIZE", 0); write = h.get("WRITE_SIZE", 0)
            short = kn.split("::")[-1].split(" (")[0].strip()
            print(f"  {short:50s} disp={n:4d} L2hit={rate:6.2f}%  HIT={hit:.0f} MISS={miss:.0f}  HBMfetch={fetch:.0f}KB HBMwrite={write:.0f}KB")

for M in sys.argv[1:]:
    parse(f"/tmp/rp2s_m{M}")
