"""Clean calibration: a single large memcpy (read src + write dst, both N bytes).
N = 256 MiB of fp32 = 268435456 elements * 4 bytes.
  floor FETCH = N*4 bytes (read src once); floor WRITE = N*4 bytes (write dst once).
Run a few copies; rocprof wraps it. Determine bytes-per-rocprof-unit."""
import torch
N = 268435456  # 256 MiB of fp32
src = torch.randn(N, device="cuda", dtype=torch.float32)
dst = torch.empty(N, device="cuda", dtype=torch.float32)
for _ in range(3):
    dst.copy_(src)
torch.cuda.synchronize()
print(f"memcpy N={N} fp32: src=dst={N*4/1e6:.0f}MB = {N*4/1e9:.3f}GB", flush=True)
print(f"floor FETCH = {N*4} bytes ; floor WRITE = {N*4} bytes", flush=True)
print("CALIB2_DONE", flush=True)
