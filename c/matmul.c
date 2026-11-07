// Naive matrix multiply: C = A x B, all N x N floats. You write the three loops.
// Build:  cc -O2 -o c/matmul c/matmul.c
// Run:    ./c/matmul
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = calloc(N * N, sizeof(float));
    for (int i = 0; i < N * N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

    double t0 = now();

    // TODO: the three loops (i, j, k).
    // Element (row r, column c) of a matrix M is M[r * N + c].
    // C[i][j] = sum over k of A[i][k] * B[k][j]
    for (int i=0; i<N; i++) {
    	for (int j=0; j<N; j++) {
		for (int k=0; k<N; k++) {
			C[i*N+j] += A[i*N+k]*B[j+k*N];
		}		

    	}   
    }
    
    double secs = now() - t0;

    printf("C[0] = %.0f (should be %d)\n", C[0], 2 * N);
    printf("time: %.3f s\n", secs);
    // TODO: print GFLOPS. A matmul does 2 * N * N * N operations; GFLOPS = operations / secs / 1e9.
    printf("GFLOPS: %.2f\n", 2.0*N*N*N/secs/1e9); 
    free(A); free(B); free(C);
    return 0;
}
