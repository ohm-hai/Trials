"""Parse phase14 rocprofv2 HBM CSV -> per-kernel FETCH/WRITE, identify fmoe kernels."""
import csv, sys, glob, os

outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rp2_p14"
csvs = sorted(glob.glob(os.path.join(outdir, "pmc_1", "results_*.csv")))
print(f"CSVs: {csvs}")
agg = {}
for c in csvs:
    with open(c) as f:
        for row in csv.DictReader(f):
            kn = row.get("Kernel_Name", "")
            try:
                fetch = float(row.get("FETCH_SIZE", 0) or 0)
                write = float(row.get("WRITE_SIZE", 0) or 0)
            except:
                fetch = write = 0.0
            d = agg.setdefault(kn, {"fetch": 0.0, "write": 0.0, "count": 0})
            d["fetch"] += fetch
            d["write"] += write
            d["count"] += 1

print(f"\n{'kernel':60s} {'disp':>5} {'FETCH_KB':>14} {'WRITE_KB':>14} {'FETCH_GB':>10}")
total_fetch = 0.0
fmoe_fetch = 0.0
fmoe_disp = 0
for kn, d in sorted(agg.items()):
    total_fetch += d["fetch"]
    short = kn.split("::")[-1].split(" (")[0].strip()[:58]
    is_fmoe = "fmoe" in kn.lower()
    if is_fmoe:
        fmoe_fetch += d["fetch"]
        fmoe_disp += d["count"]
    print(f"{short:60s} {d['count']:>5} {d['fetch']:>14.0f} {d['write']:>14.0f} {d['fetch']/1e6:>10.3f}")
print(f"\nTotal FETCH (all kernels): {total_fetch/1e6:.3f} GB")
print(f"fmoe kernels: disp={fmoe_disp} FETCH={fmoe_fetch/1e6:.3f} GB "
      f"per-dispatch={fmoe_fetch/fmoe_disp/1e6:.4f} GB")
