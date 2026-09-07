import torch, triton, triton.language as tl, time
print('cuda ok', torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
y = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
z = torch.empty(512, 512, device='cuda', dtype=torch.bfloat16)

@triton.jit
def k(xp, yp, zp, N: tl.constexpr):
    pid = tl.program_id(0)
    r = pid // N
    c = pid % N
    a = tl.load(xp + r * 512 + tl.arange(0, 512))
    b = tl.load(yp + c * 512 + tl.arange(0, 512))
    tl.store(zp + r * 512 + tl.arange(0, 512), a + b)

t = time.time()
k[(512,)](x, y, z, 512)
torch.cuda.synchronize()
print('simple triton ok in', round(time.time() - t, 3), 's', flush=True)
t = time.time()
for _ in range(5):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
print('matmul ok in', round((time.time() - t) / 5, 3), 's', flush=True)
