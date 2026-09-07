"""Calibrate rocprofv2 FETCH_SIZE/WRITE_SIZE units with a known workload:
element-wise vector add  C = A + B  (no tiling, each element read/written once).
N = 100_000_000 fp32 elements -> A=400MB, B=400MB, C=400MB.
  floor FETCH = A+B = 800_000_000 bytes; WRITE = C = 400_000_000 bytes.
If rocprof reports FETCH_SIZE ~ 8e8  -> units are BYTES.
If rocprof reports FETCH_SIZE ~ 8e5  -> units are KB (8e8 bytes)."""
import torch
N = 100_000_000
A = torch.randn(N, device="cuda", dtype=torch.float32)
B = torch.randn(N, device="cuda", dtype=torch.float32)
C = torch.empty(N, device="cuda", dtype=torch.float32)
for _ in range(3):
    torch.add(A, B, out=C)
torch.cuda.synchronize()
print(f"vector add N={N} fp32: A=B=C={N*4/1e6:.0f}MB", flush=True)
print(f"floor FETCH = {2*N*4} bytes = {2*N*4/1e9:.2f}GB", flush=True)
print(f"floor WRITE = {N*4} bytes = {N*4/1e9:.2f}GB", flush=True)
print("CALIB_DONE", flush=True)
