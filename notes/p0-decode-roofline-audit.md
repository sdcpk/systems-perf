Claim:
Autoregressive decode at batch 1 wastes most GPU FLOPs because it is memory-bandwidth-bound.

Sources:
1. NVIDIA lists the RTX 4090 with 24 GB GDDR6X memory; use public RTX 4090 specs for memory configuration and bandwidth.
2. NVIDIA Ada architecture materials list RTX 4090 dense FP16 tensor throughput with FP32 accumulation as 165.2 TFLOP/s.
3. Meta’s Llama 3.1 model card lists the Llama 3.1 collection in 8B, 70B, and 405B sizes.

Restated as a number:
On an RTX 4090, dense bf16/fp16 batch-1 decode has AI ≈ 1 FLOP/byte under a weight-streaming model. The RTX 4090 ridge point is ≈164 FLOPs/byte using 165.2 TFLOP/s dense bf16/fp16 tensor compute and 1.008 TB/s memory bandwidth. Therefore batch-1 decode has a roofline utilization ceiling of ≈1/164 = 0.61% of peak FLOP/s, meaning ≈99.4% of peak FLOP/s is unavailable under this model.

Inputs:
1. GPU peak bf16/fp16 tensor compute: 165.2 TFLOP/s
2. GPU memory bandwidth: 1008 GB/s = 1.008 TB/s GDDR6X bandwidth
3. Precision: bf16/fp16 weights = 2 bytes/parameter
4. Batch size: B
5. Dense model parameter count: P
6. Model: Llama 3.1 8B, so P ≈ 8e9

Assumptions:
1. Dense transformer decode.
2. Weights are streamed from GPU memory once per decode step.
3. Weights are reused across batch B.
4. Weight traffic dominates.
5. Ignores KV-cache traffic, activations, sampling, launch overhead, cache effects, quantization, sparsity, and imperfect kernel utilization.
6. Uses theoretical dense tensor-core peak, not measured achieved FLOP/s.

Derivation:
FLOPs per batch decode step ≈ 2PB
Bytes per batch decode step ≈ 2P
AI ≈ 2PB / 2P = B FLOPs/byte

For B=1:
AI ≈ 1 FLOP/byte

RTX 4090 ridge:
R = 165.2 TFLOP/s / 1.008 TB/s ≈ 164 FLOPs/byte

Utilization ceiling:
U(B) ≈ min(B/R, 1)

For B=1:
U(1) ≈ 1/164 ≈ 0.61%

For Llama 3.1 8B:
FLOPs per generated token at B=1 ≈ 2 * 8e9 = 16e9 FLOPs
Weight bytes per generated token at B=1 ≈ 2 * 8e9 = 16 GB

Boundary:
Wastes most FLOPs when U(B) < 50%, so B < R/2 ≈ 82.
Memory-bound when B < R ≈ 164.
Compute-bound only once B approaches or exceeds ≈164, under the weight-streaming model.

Result as condition:
Under a dense bf16/fp16 weight-streaming model on RTX 4090, the claim is true at batch 1: autoregressive decode has AI ≈1 FLOP/byte and a theoretical utilization ceiling of ≈0.61% of peak FLOP/s. The claim remains true in the “wastes most FLOPs” sense for batch sizes below roughly 82, and decode remains memory-bound until batch size approaches roughly 164.