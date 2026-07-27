1. Arithmetic intensity of large bf16 matmul.
FLOPS = 2DBF, bytes = 2 * (BF + DF + DB)
AI = BDF/(BF+DF+DB)
2. Arithmetic intensity of attention at sequence length 4096.
FLOPS = 4*(4096^2*dhead)
//4 matrices for input and output Q,K,V,O +
//write S, read S, write P, read P
Memory = 2*4*(4096*head) + 4(4069^2) + 4(4096^2)
AI = 4*(4096^2*dhead)/(8*4096*dhead+8*4096^2)
3. Arithmetic intensity of autoregressive decode at batch 1.
If P is total params then batch 1 (or a 1xM) matrix will multiply several MxN matrices. If each param is a MxN matrix Pi, the the sum of the mat muls will be 2P FLOPS where P is the the sum of the MxN param matrices. 
We have to move 2P bytes. 2P/2P gives us 1FLOP/byte
4. Arithmetic intensity of autoregressive decode at batch 256.
We will have a 256xM matrix multiplying MxN matrices. Each of those operations will be 2*256*M*N. This is for each Pi=MxN. So flops equals 2*256*P. Bytes moved is 2*P, so AI=256. 
5. Ridge point of the GPU you are renting, using real public specs.
RTX 4090 has 165.2 TFLOPS
Memory 1TB/s
so ridgepoint is 165FLOPS/byte