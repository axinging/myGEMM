
// =================================================================================================
// Simple CPU SGEMM used as the 'golden' reference when neither cuBLAS nor clBlas is available.
// Column-major, no transposes, alpha=1, beta=0 (same layout as the myGEMM kernels).
// =================================================================================================

#include "common.h"

void libcpu(float* A, float* B, float* C,
            int K, int M, int N) {
    for (int n=0; n<N; n++) {
        float* c = &C[(size_t)n*M];
        for (int m=0; m<M; m++) { c[m] = 0.0f; }
        for (int k=0; k<K; k++) {
            const float b = B[(size_t)n*K + k];
            const float* a = &A[(size_t)k*M];
            for (int m=0; m<M; m++) {
                c[m] += a[m] * b;
            }
        }
    }
}

// =================================================================================================
