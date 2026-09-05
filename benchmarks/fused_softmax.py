# Goal: benchmark row-wise fused softmax against torch.softmax.
#
# Shape:
#   X has shape [M, N].
#   First target: M = 4096, N = 4096, dtype = fp32.
#
# Mapping:
#   One Triton program handles one row.
#   program_id(0) gives row_id.
#   cols = 0..BLOCK_SIZE-1.
#   offsets = row_id * N + cols.
#
# Softmax per row:
#   m = max(x)
#   num = exp(x - m)
#   den = sum(num)
#   y = num / den
#
# Benchmark metrics:
#   median time, p90/std, estimated bytes, effective bandwidth,
#   speedup vs torch, max error vs torch.

import statistics
import subprocess

import torch
import triton
import triton.language as tl


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, (90 * len(xs)) // 100)]


def driver_version():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return "unknown"

@triton.jit
def fused_softmax_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row_id * N + cols

    x = tl.load(x_ptr + offsets, mask=cols < N, other=float("-inf"))
    m = tl.max(x, axis=0)
    num = tl.exp(x - m)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(y_ptr + offsets, y, mask=cols < N)


M, N = 4096, 4096
WARMUP = 10
REPEATS = 50

# Fused traffic: read X + write Y = 8 * MN = 128 MiB for M = N = 4096.
FUSED_BYTES = 128 * (1 << 20)

x = torch.randn((M, N), device="cuda", dtype=torch.float32)
# dim=-1 == dim=1 for 2D [M, N]: softmax over columns, i.e. each row.
y_ref = torch.softmax(x, dim=-1)

y = torch.empty_like(x)
BLOCK_SIZE = triton.next_power_of_2(N)
fused_softmax_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
torch.cuda.synchronize()
max_error = (y - y_ref).abs().max().item()
print(f"triton max error vs ref: {max_error:.3e}")

for _ in range(WARMUP):
    torch.softmax(x, dim=-1)
torch.cuda.synchronize()

times_ms_torch = []
for _ in range(REPEATS):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.softmax(x, dim=-1)
    end.record()
    torch.cuda.synchronize()
    times_ms_torch.append(start.elapsed_time(end))

torch_median_ms = statistics.median(times_ms_torch)
torch_p90_ms = p90(times_ms_torch)
torch_std_ms = statistics.pstdev(times_ms_torch)
torch_seconds = torch_median_ms / 1e3
torch_bw = FUSED_BYTES / torch_seconds / 1e9

print(f"torch median time: {torch_median_ms:.3f} ms")
print(f"torch estimated bytes: {FUSED_BYTES / (1 << 20):.0f} MiB")
print(f"torch effective bandwidth: {torch_bw:.1f} GB/s")

for _ in range(WARMUP):
    fused_softmax_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
torch.cuda.synchronize()

times_ms_triton = []
for _ in range(REPEATS):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fused_softmax_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    times_ms_triton.append(start.elapsed_time(end))

triton_median_ms = statistics.median(times_ms_triton)
triton_p90_ms = p90(times_ms_triton)
triton_std_ms = statistics.pstdev(times_ms_triton)
triton_seconds = triton_median_ms / 1e3
triton_bw = FUSED_BYTES / triton_seconds / 1e9

print(f"triton median time: {triton_median_ms:.3f} ms")
print(f"triton effective bandwidth: {triton_bw:.1f} GB/s")
print(f"speedup vs torch: {torch_median_ms / triton_median_ms:.2f}x")

print("\n=== run record ===")
print(f"GPU model: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"driver version: {driver_version()}")
print(f"M, N: {M}, {N}")
print(f"dtype: {x.dtype}")
print(f"warmup count: {WARMUP}")
print(f"repeat count: {REPEATS}")
print(f"torch median / p90 / std: {torch_median_ms:.3f} / {torch_p90_ms:.3f} / {torch_std_ms:.3f} ms")
print(f"triton median / p90 / std: {triton_median_ms:.3f} / {triton_p90_ms:.3f} / {triton_std_ms:.3f} ms")
print(f"max error: {max_error:.3e}")
