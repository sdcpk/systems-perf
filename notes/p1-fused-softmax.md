## Goal
Explain why this fused softmax kernel is memory system limited and compare its performance against pytorch softmax.

## Kernel mechanism
The kernel program loads one row for each program id. The program computes the row max, shifts each element by the max, exponentiates, and normalizes and writes output. The fused kernel does not explicitly write the shifted values or exponentials out as separate global tensors. Its minimum traffic model is therefore one read of X and one write of Y, or 8MN bytes for fp32.

## Byte model
In the byte model, with X in R^(MxN) using fp32 elements, we have:
```
read X, write shifted (i.e. X-m), read shifted, write exp, read exp, write Y
```
bytes = 6*4(MN)
for fused version of softmax it's read X, write Y, which is 8MN and for 4096x4096 fp32 that's the 128MiB model for bandwidth

A decomposed softmax that materializes the shifted and exponential tensors would require approximately 24MN bytes of tensor traffic under this simplified model. The fused Triton kernel has a minimum explicit global-traffic model of 8MN bytes: one read of X and one write of Y.

## Benchmark methodology
GPU: NVIDIA RTX A5000
PyTorch: 2.4.1+cu124
Triton: 3.0.0
CUDA: 12.4
Driver: 570.195.03

M=N=4096
dtype=float32
warmup=10
repeats=50

Benchmark ran on an RTX A5000 with M=N=4096 fp32.
Used 10 warmup iterations and 50 timed repetitions.
CUDA events measured GPU execution time, with synchronization around timing.
Reported median, p90, standard deviation, speedup vs torch.softmax,
and max absolute error.

## A5000 results
Torch: 0.213 ms median
Triton: 0.207 ms median
Speedup: ~1.03x
Max error: 1.863e-09
Torch p90/std: 0.228 / 0.006 ms
Triton p90/std: 0.213 / 0.003 ms
Estimated minimum traffic: 128 MiB

Torch effective bandwidth: 629.2 GB/s
Triton effective bandwidth: 647.2 GB/s


## Nsight Compute methodology
Profiling was performed separately on an H100 PCIe because the A5000
cloud container did not expose NVIDIA hardware performance counters.
Nsight Compute 2025.3.1 profiled one fused_softmax_kernel launch using
the full metric set. Benchmark timings and profiler measurements therefore
come from different GPUs and are not directly compared.

## H100 profiler results
DRAM throughput: 84.23%
Compute throughput: 25.21%
L2 throughput: 81.46%
Registers/thread: 54
Theoretical occupancy: 56.25%
Achieved occupancy: 51.43%

## Interpretation
DRAM throughput was 84.23% while compute throughput was only 25.21%, indicating that the memory system was much more heavily utilized than the compute pipelines. 54 registers/thread limits theoretical occupancy to 56.25%.Even with 51.43% achieved occupancy, there were enough outstanding memory requests to drive DRAM throughput to 84.23%. So just increasing occupancy might not result in a faster kernel.

## Limitations
H100 profile does not explain the precise A5000 timing because they're different architectures/environments, and 128 MiB is a modeled minimum traffic amount rather than measured DRAM bytes.

## Conclusion
The Triton implementation reached near parity with PyTorch softmax on the A5000, eliminating explicit intermediate global tensors. Nsight profiling on an H100 showed the kernel pushing the memory system much more than compute, while register usage limited occupancy. The profile also showed that lower than maximum occupancy can still be sufficient to drive high memory bandwidth.