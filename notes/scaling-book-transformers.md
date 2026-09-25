## Questions

Question 1: How many parameters does a model with D = 4 0 9 6 D=4096, F = 4 ⋅ D F=4⋅D, V = 3 2 , 0 0 0 V=32,000, and L = 6 4 L=64 have? What fraction of these are attention parameters? How large are our KV caches per token? You can assume N ⋅ H = D N⋅H=D and multi-head attention with int8 KVs.

1. PARAMS = L*(2DNH + 2DKH + 3DF + 2D) + 2DV = L*(4DNH + 3DF + 2D) + 2DV
64*(4*4e3*4e3 + 3*4e3*16e3 + 2*4e3) + 2*4e3*32000 
=64*(4*16e6 + 3*64e6 + 8e3) + 256e6 = 4096e6 + 12288e6 + 512e3 + 256e6
=16640.512e6
2. (4DNH)/(4DNH + 3DF) = 1/4 with F=4D and D=NH
3. Our KV caches per token are 2LNH

Question 2: How many total FLOPs are required to perform A[BX, DY] *D W[DY, F] on {'X': 4, 'Y': 8, 'Z': 4}? How many FLOPs are performed by each TPU?
1. [Bx/4,Dy/8]*[Dy/8,F] = [Bx/4,F] in each TPU, 2*(Bx/4)*(Dy/8)*F=BDF/16 flops for each TPU
mesh has 4*8*4 TPUs so 128BDF/16= 8BDF

Question 3: How many FLOPs are involved in performing A[I,J,K,L]*B[I,J,M,N,O]→C[K,L,M,N,O]
2IJKLMNO FLOPS

Question 4: What is the arithmetic intensity of grouped multi-query attention (ignoring the Q/K/V/O projections)? Give the answer as a function of the Q and KV sequence lengths T and S and the multi-query factor G. At what context length is attention FLOPs-bound? Given the HBM bandwidth of our TPUs, plot the effective relative cost of attention to the FFW block as the context length grows. Hint: assume we’re using an efficient attention implementation that doesn’t do any unnecessary reads/writes. Consider both the limits where T = S and T « S.

Read Q,K,V for forward pass is 2*sizeof(Q)+2*sizeof(K or V), O is same size as Q
4BTNH + 4BSKH = 4BHK*(TG+S)
Q[B,T,K,G,H]*K[B,S,K,H] = [B,T,S,K,G,H] is 2BTSKGH FLOPS
[B,T,K,G,H]*V[B,S,K,H] = 2BTSKGH FLOPS
softmax flops is size([B,T,S,K,G,H])
total flops is 4BTSKGH, ignoring softmax (smaller by 1/H factor)
Arithmetic intensity is 4BTSKGH/4BHK*(TG+S) 
There's prefill and decode, during prefill T=S so we have 4BT^2KGH/4BHKT(G+1) = TG/(G+1)
=O(T)
In decode T=1, so we have SG/(G+S) =~ G for large S

Question 5: At what sequence length are self-attention FLOPs equal to the QKVO projection FLOPs?
24BTDNH = 12BTSKGH 
for T=K, 2D=T so D=T/2

Question 6: Say we only save the output of each of the 7 main matmuls in a Transformer layer during our forward pass (Q, K, V, O + the three FFW matrices). How many extra FLOPs do we need to “rematerialize” during the backward pass?

During backward pass we need dL/dW and dL/dx 
dL/dx = dL/dY * W^T 
dL/dW= X^T*dL/dY
But A=QK^T and O=PV do not have learned weights, so we need to rematerialize
2BT^2KGH + 2BT^2KGH = 4BT^2KGH FLOPS

Question 7: DeepSeek v3 says it was trained for 2.79M H800 hours on 14.8T tokens (source). Given that it has 37B activated parameters, roughly what hardware utilization did they achieve? Hint: note that they used FP8 FLOPs without structured sparsity.

3,026 TFLOPs/s of FP8 performance with sparsity, or typically half this (1.513e15 FLOPs/s) without sparsity.
1.513e15*2.79e6*60*60=1.52e25 FLOPS used in training
2N flops per token in forward pass and 4N in backward so 6N per token. 
6 * 37e9 * 14.8e12 = 3.3e24

Question 8: Mixture of Experts (MoE) models have E E copies of a standard dense MLP block, and each token activates k k of these experts. What batch size in tokens is required to be compute-bound for an MoE with weights in int8 on TPU v5e? For DeepSeek, which has 256 (routed) experts and k = 8 k=8, what is this number?
EDF bytes loaded since there are E copies
2kBDF FLOPS 
to be compute bound 2kB/E should be higher than the hardware limit (for TPU v5e >240)

For deepseek we need B > 120E/k
B>3850


