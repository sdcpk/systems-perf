First pass — incorrect attempt kept for diagnosis
1. Flops for 4096 x 4096 bf matmul
nxn * nxn = nxn
for each element we have n multiplies and n-1 adds
so n^2*(2n-1)
for n is 4096=2^24*(2^13-1) = 2^11 FLOPS

bytes will be 2(2^12 + 2^12 + 2^12) = 2^15 bytes

2. The ridge point is the point on a graph where the hardware speed matches its memory loading speed.
100 TFLOP/s, 2 TB/s, AI=20FLOPS/byte
the hardware intensity is 50FLOPS/byte

3. Q,K,V have shape BxTxd where B is batch size, T is tokens, and d is feature dimension
QK^T has shape BxTxT, P has shape BxTxT and PV has shape BxTxT, PV has shape BxTxd

4. X -> QKV -> Q*K^T -> causal mask -> P=softmax(Q*K^T) -> PV -> residual add

5. A thread is an individual worker. A warp is a group of threads. A block/CTA is a group of threads with shared memory. An SM is a hardware unit that schedules blocks. Registers are indivudal to a thread, shared memory is memory space for a block, and HBM is high bandwidth memory off chip. 

Second pass corrections: 1. 2^37-2^24 FLOPS, 3*2^25bytes, 2^37 FLOPS / 3*2^25 bytes = 2^12/3 FLOPS/byte 2. 100 TFLOP/s,2 TB/s,AI=20 Ridge point = 50FLOPS/s, AI<Ridge point so performance = 20FLOPS/byte * 2TB/s = 40 TFLOPS/s 3. Q,K,V have shape BxTxd where B is batch size, T is tokens, and d is feature dimension QK^T has shape BxTxT, P has shape BxTxT and PV has shape BxTxd 4. When you said prenorm i thought you mean exclude layer norm. X -> LayerNorm(X) -> X*W -> Q,K,V -> Q*K^T -> causal mask -> softmax(Q*K^T) = P -> PV -> PV * Wo -> Residual Add X -> LayerNorm -> MLP -> Residual Add with X

1. LayerNorm operates over the feature dimension d
2. LayerNorm does not mix info between tokens because it normalize one token's feature dimension.
3. The X input is bring added to the output of the attention block.
4. Residual add is useful because Y = X + F(X), so information and gradients have a direct identity path around the sublayer while the sublayer learns an update to X.
Pre-norm decoder block: X1 = X + Attn(LN(X)); X2 = X1 + MLP(LN(X1)).

Thread = one logical unit of execution with its own registers.

Warp = usually 32 threads executing the same instruction together.

Block/CTA = a group of threads that can share shared memory and synchronize.

SM = the hardware unit that schedules and executes warps from resident blocks.

Register = fast thread-private storage.

Shared memory = fast on-chip memory shared by threads in one block.

L2 = hardware-managed on-chip cache shared across SMs.

HBM = large off-chip global memory with high bandwidth but high latency.

Coalescing = adjacent threads in a warp access adjacent addresses, allowing efficient memory transactions.

Latency hiding = the SM runs other ready warps while one warp waits on memory.


Coalescing and latency hiding examples
1. If thread 0 loads x[0], thread 1 loads x[1], ..., thread 31 loads x[31], is this coalesced? Why?
Yes. Threads in the same warp access consecutive addresses, so the hardware can combine the load into fewer memory transactions.

2. If thread 0 loads x[0], thread 1 loads x[1000], thread 2 loads x[2000], ..., is this coalesced? Why?
No. This is not coalesced because adjacent threads in the warp access far-apart addresses, so the hardware must issue scattered memory transactions instead of one/few contiguous transactions.

3. In a pointer chain where each load gives the address of the next load, why is latency hiding harder?
Pointer chasing makes latency hiding harder because each load’s address is unknown until the previous load completes, so there is little memory-level parallelism within that chain.

A warp has 32 threads. Each thread loads one bf16 value.

Case A: lane i loads x[i].
Case B: lane i loads x[1024*i].
Case C: lane i loads x[base + i], but base changes each loop iteration.

For each case:
1. coalesced or not?
2. why?
3. what happens to memory transactions?

Case A
Yes, coalesced. Neighboring lanes access neighboring bf16 elements, so the warp presents one contiguous region to memory. Result: fewer, efficient memory transactions.

Case B Not coalesced. Adjacent lanes access addresses far apart from each other. Result: many scattered memory transactions, worse bandwidth use.

Case C is coalesced if, within each loop iteration, lane i loads x[base + i]. Coalescing is judged across warp lanes for the same memory instruction. Changing base across loop iterations does not break coalescing unless the next base depends on a previous memory load, which creates dependent loads and hurts latency hiding.

1. Coalesced load pattern:
lane i loads x[i]. This is coalesced because adjacent lanes access adjacent addresses. Problem: neither bandwidth transactions nor latency dependency, assuming the addresses are known and aligned.

2. Non-coalesced strided load pattern:
lane i loads x[i*1000]. This is not coalesced because adjacent lanes access far-apart addresses. Problem: inefficient bandwidth transactions.

3. Coalesced but latency-dependent pattern:
lane i loads x[base + i], so each iteration is coalesced. But if the next base is loaded from memory and depends on the previous iteration, the warp cannot issue the next load until that base returns. Problem: latency hiding / lack of memory-level parallelism, not coalescing.

Final diagnostic summary

My main mistakes were:
1. I collapsed tensor dimensions instead of preserving them.
2. I confused arithmetic intensity units with throughput units.
3. I added tensor dimensions when counting bytes instead of multiplying dimensions.
4. I did not initially distinguish coalescing from latency hiding.
5. I had GPU vocabulary but not the mechanism.

Correct rules:
1. Matmul FLOPs come from output elements times work per output element.
2. Tensor bytes come from multiplying all dimensions, then multiplying by bytes per element.
3. Ridge point = peak FLOP/s divided by peak byte/s, with units FLOPs/byte.
4. Attention preserves the query-token dimension.
5. Coalescing is about adjacent warp lanes accessing adjacent addresses for one memory instruction.
6. Latency hiding is about whether the SM has independent ready work while a warp waits on memory.