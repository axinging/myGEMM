// =================================================================================================
// SYCL backend of myGEMM: a 1:1 port of all OpenCL kernels (opencl/kernels.cl, KERNEL 1 -- 11),
// including the supporting transpose / padding kernels. Same data layout, same tiling constants
// (src/settings.h), same sequence of kernels and same timing method as opencl/clGEMM.cpp, so that
// correctness and performance can be compared 1:1 with the OpenCL backend.
//
// Dimension mapping: OpenCL dimension 0 (the fastest varying one) is SYCL dimension 1 of an
// nd_range<2>, hence the reversed ranges (ndr) and the L0/L1/G0/G1/GL0/GL1 helpers below.
// =================================================================================================

// Note: the SYCL header has to come before settings.h (which defines short macros like MIN/TS)
#include <sycl/sycl.hpp>

#include "common.h"
#include "settings.h"

typedef sycl::vec<float, WIDTH> floatX;

// OpenCL-style work-item functions (OpenCL dim d  ==  SYCL dim 1-d)
#define L0  ((int)it.get_local_id(1))   // get_local_id(0)
#define L1  ((int)it.get_local_id(0))   // get_local_id(1)
#define G0  ((int)it.get_group(1))      // get_group_id(0)
#define G1  ((int)it.get_group(0))      // get_group_id(1)
#define GL0 ((int)it.get_global_id(1))  // get_global_id(0)
#define GL1 ((int)it.get_global_id(0))  // get_global_id(1)
#define BARRIER() sycl::group_barrier(it.get_group())

// nd_range from OpenCL-style (dim0, dim1) global/local sizes
static sycl::nd_range<2> ndr(size_t g0, size_t g1, size_t l0, size_t l1) {
    return sycl::nd_range<2>(sycl::range<2>(g1, g0), sycl::range<2>(l1, l0));
}
// Round up to a multiple of l (used for the supporting kernels, which bounds-check themselves)
static size_t up(int x, int l) { return (size_t)CEIL_DIV(x, l) * l; }

// =================================================================================================

void mysycl(float* A, float* B, float* C,
            int K, int M, int N,
            int timerID) {

    // In case of myGEMM10, compute matrix sizes K, M, N as rounded-up to form complete tiles
    #if KERNEL == 10
        const int K_XL = CEIL_DIV(K, TSK) * TSK;
        const int M_XL = CEIL_DIV(M, TSM) * TSM;
        const int N_XL = CEIL_DIV(N, TSN) * TSN;
    #else
        const int K_XL = K;
        const int M_XL = M;
        const int N_XL = N;
    #endif

    // Select the GPU; ONEAPI_DEVICE_SELECTOR can be used to pick the SYCL runtime backend
    // (e.g. ONEAPI_DEVICE_SELECTOR=opencl:gpu  or  level_zero:gpu)
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    static bool printed = false;
    if (!printed) {
        sycl::device d = q.get_device();
        printf("## SYCL device: '%s' (SYCL backend id: %d)\n",
               d.get_info<sycl::info::device::name>().c_str(), (int)d.get_backend());
        printed = true;
    }

    // Device memory
    float* dA    = sycl::malloc_device<float>((size_t)M*K, q);
    float* dB    = sycl::malloc_device<float>((size_t)K*N, q);
    float* dB_TR = sycl::malloc_device<float>((size_t)N*K, q);
    float* dC    = sycl::malloc_device<float>((size_t)M*N, q);
    float* dA_XL    = sycl::malloc_device<float>((size_t)M_XL*K_XL, q);
    float* dB_TR_XL = sycl::malloc_device<float>((size_t)N_XL*K_XL, q);
    float* dC_XL    = sycl::malloc_device<float>((size_t)M_XL*N_XL, q);
    if (!dA || !dB || !dB_TR || !dC || !dA_XL || !dB_TR_XL || !dC_XL) {
        printf("-- Error: SYCL allocation failed\n");
        exit(1);
    }

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    q.memcpy(dA, A, (size_t)M*K*sizeof(float));
    q.memcpy(dB, B, (size_t)K*N*sizeof(float));
    q.memcpy(dC, C, (size_t)M*N*sizeof(float));
    q.wait();

    // Which buffers go into the myGEMM kernel
    #if KERNEL == 10
        const float* gA = dA_XL;
        const float* gB = dB_TR_XL;
        float* gC = dC_XL;
    #elif KERNEL >= 5 && KERNEL <= 9
        const float* gA = dA;
        const float* gB = dB_TR;
        float* gC = dC;
    #else
        const float* gA = dA;
        const float* gB = dB;
        float* gC = dC;
    #endif

    // One complete run: [transpose] [pad] myGEMM [unpad]
    auto run = [&]() {

        // ---- Supporting transpose kernel: B (K*N) -> B_TR (N*K) --------------------------------
        #if KERNEL >= 5 && KERNEL <= 10
        {
            const int P = K, Q = N;
            const float* in = dB;
            float* out = dB_TR;
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> buffer(sycl::range<2>(TRANSPOSEX, TRANSPOSEY), h);
                h.parallel_for(ndr(up(P,TRANSPOSEX), up(Q,TRANSPOSEY), TRANSPOSEX, TRANSPOSEY),
                               [=](sycl::nd_item<2> it) {
                    const int tx = L0;
                    const int ty = L1;
                    const int ID0 = G0*TRANSPOSEX + tx; // 0..P
                    const int ID1 = G1*TRANSPOSEY + ty; // 0..Q
                    if (ID0 < P && ID1 < Q) {
                        buffer[ty][tx] = in[ID1*P + ID0];
                    }
                    BARRIER();
                    const int newID0 = G1*TRANSPOSEY + tx;
                    const int newID1 = G0*TRANSPOSEX + ty;
                    if (newID0 < Q && newID1 < P) {
                        out[newID1*Q + newID0] = buffer[tx][ty];
                    }
                });
            });
        }
        #endif

        // ---- Supporting padding kernels (myGEMM10 only) ----------------------------------------
        #if KERNEL == 10
        {
            auto padAdd = [&](int P, int Q, const float* in, int P_XL, int Q_XL, float* out) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(ndr(up(P_XL,PADDINGX), up(Q_XL,PADDINGY), PADDINGX, PADDINGY),
                                   [=](sycl::nd_item<2> it) {
                        const int tx = G0*PADDINGX + L0;
                        const int ty = G1*PADDINGY + L1;
                        if (tx < P_XL && ty < Q_XL) {
                            float value = (tx < P && ty < Q) ? in[ty*P + tx] : 0.0f;
                            out[ty*P_XL + tx] = value;
                        }
                    });
                });
            };
            padAdd(M, K, dA, M_XL, K_XL, dA_XL);
            padAdd(N, K, dB_TR, N_XL, K_XL, dB_TR_XL);
        }
        #endif

        // ---- The myGEMM kernel -----------------------------------------------------------------
        {
            // Inside this block M, N, K are the (possibly rounded-up) sizes the kernel works on
            const int M = M_XL, N = N_XL, K = K_XL;
            const float* pA = gA;
            const float* pB = gB;
            float* pC = gC;

            // ================================ KERNEL 1 =========================================
            #if KERNEL == 1
            q.submit([&](sycl::handler& h) {
                h.parallel_for(ndr(M, N, TS, TS), [=](sycl::nd_item<2> it) {
                    const int globalRow = GL0;
                    const int globalCol = GL1;
                    float acc = 0.0f;
                    for (int k=0; k<K; k++) {
                        acc += pA[k*M + globalRow] * pB[globalCol*K + k];
                    }
                    pC[globalCol*M + globalRow] = acc;
                });
            });

            // ================================ KERNEL 2 =========================================
            #elif KERNEL == 2
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(TS, TS), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(TS, TS), h);
                h.parallel_for(ndr(M, N, TS, TS), [=](sycl::nd_item<2> it) {
                    const int row = L0;
                    const int col = L1;
                    const int globalRow = TS*G0 + row;
                    const int globalCol = TS*G1 + col;
                    float acc = 0.0f;
                    const int numTiles = K/TS;
                    for (int t=0; t<numTiles; t++) {
                        const int tiledRow = TS*t + row;
                        const int tiledCol = TS*t + col;
                        Asub[col][row] = pA[tiledCol*M + globalRow];
                        Bsub[col][row] = pB[globalCol*K + tiledRow];
                        BARRIER();
                        for (int k=0; k<TS; k++) {
                            acc += Asub[k][row] * Bsub[col][k];
                        }
                        BARRIER();
                    }
                    pC[globalCol*M + globalRow] = acc;
                });
            });

            // ================================ KERNEL 3 =========================================
            #elif KERNEL == 3
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(TS, TS), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(TS, TS), h);
                h.parallel_for(ndr(M, N/WPT, TS, TS/WPT), [=](sycl::nd_item<2> it) {
                    const int row = L0;
                    const int col = L1;
                    const int globalRow = TS*G0 + row;
                    const int globalCol = TS*G1 + col;
                    float acc[WPT];
                    for (int w=0; w<WPT; w++) { acc[w] = 0.0f; }
                    const int numTiles = K/TS;
                    for (int t=0; t<numTiles; t++) {
                        for (int w=0; w<WPT; w++) {
                            const int tiledRow = TS*t + row;
                            const int tiledCol = TS*t + col;
                            Asub[col + w*RTS][row] = pA[(tiledCol + w*RTS)*M + globalRow];
                            Bsub[col + w*RTS][row] = pB[(globalCol + w*RTS)*K + tiledRow];
                        }
                        BARRIER();
                        for (int k=0; k<TS; k++) {
                            for (int w=0; w<WPT; w++) {
                                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
                            }
                        }
                        BARRIER();
                    }
                    for (int w=0; w<WPT; w++) {
                        pC[(globalCol + w*RTS)*M + globalRow] = acc[w];
                    }
                });
            });

            // ================================ KERNEL 4 =========================================
            #elif KERNEL == 4
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<floatX, 2> Asub(sycl::range<2>(TS, TS/WIDTH), h);
                sycl::local_accessor<floatX, 2> Bsub(sycl::range<2>(TS, TS/WIDTH), h);
                h.parallel_for(ndr(M/WIDTH, N, TS/WIDTH, TS), [=](sycl::nd_item<2> it) {
                    const floatX* A4 = reinterpret_cast<const floatX*>(pA);
                    const floatX* B4 = reinterpret_cast<const floatX*>(pB);
                    floatX* C4 = reinterpret_cast<floatX*>(pC);
                    const int row = L0;
                    const int col = L1;
                    const int globalRow = (TS/WIDTH)*G0 + row;
                    const int globalCol = TS*G1 + col;
                    floatX acc(0.0f);
                    const int numTiles = K/TS;
                    for (int tile=0; tile<numTiles; tile++) {
                        const int tiledRow = (TS/WIDTH)*tile + row;
                        const int tiledCol = TS*tile + col;
                        Asub[col][row] = A4[tiledCol*(M/WIDTH) + globalRow];
                        Bsub[col][row] = B4[globalCol*(K/WIDTH) + tiledRow];
                        BARRIER();
                        for (int k=0; k<TS/WIDTH; k++) {
                            floatX vecB = Bsub[col][k];
                            for (int w=0; w<WIDTH; w++) {
                                floatX vecA = Asub[WIDTH*k + w][row];
                                float valB = vecB[w];
                                acc += vecA * valB;
                            }
                        }
                        BARRIER();
                    }
                    C4[globalCol*(M/WIDTH) + globalRow] = acc;
                });
            });

            // ================================ KERNEL 5 =========================================
            #elif KERNEL == 5
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(TSDK, TS), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(TS, TSDK+2), h);
                h.parallel_for(ndr(M, N/WPT, TS, TS/WPT), [=](sycl::nd_item<2> it) {
                    const int row = L0;
                    const int col = L1;
                    const int globalRow = TS*G0 + row;
                    const int globalCol = TS*G1 + col;
                    float acc[WPT];
                    for (int w=0; w<WPT; w++) { acc[w] = 0.0f; }
                    const int numTiles = K/TSDK;
                    for (int t=0; t<numTiles; t++) {
                        for (int l=0; l<LPT; l++) {
                            const int tiledIndex = TSDK*t + col + l*RTS;
                            int indexA = (tiledIndex)*M + TS*G0 + row;
                            int indexB = (tiledIndex)*N + TS*G1 + row;
                            Asub[col + l*RTS][row] = pA[indexA];
                            Bsub[row][col + l*RTS] = pB[indexB];
                        }
                        BARRIER();
                        for (int k=0; k<TSDK; k++) {
                            for (int w=0; w<WPT; w++) {
                                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
                            }
                        }
                        BARRIER();
                    }
                    for (int w=0; w<WPT; w++) {
                        pC[(globalCol + w*RTS)*M + globalRow] = acc[w];
                    }
                });
            });

            // ================================ KERNEL 6 =========================================
            #elif KERNEL == 6
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(TSK, TSM), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(TSN, TSK+2), h);
                h.parallel_for(ndr(M/WPTM, N/WPTN, TSM/WPTM, TSN/WPTN), [=](sycl::nd_item<2> it) {
                    const int tidm = L0;
                    const int tidn = L1;
                    const int offsetM = TSM*G0;
                    const int offsetN = TSN*G1;
                    float Breg[WPTN];
                    float acc[WPTM][WPTN];
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) { acc[wm][wn] = 0.0f; }
                    }
                    const int numTiles = K/TSK;
                    int t=0;
                    do {
                        #pragma unroll
                        for (int la=0; la<LPTA; la++) {
                            int tid = tidn*RTSM + tidm;
                            int id = la*RTSN*RTSM + tid;
                            int row = MOD2(id,TSM);
                            int col = DIV2(id,TSM);
                            int tiledIndex = TSK*t + col;
                            Asub[col][row] = pA[tiledIndex*M + offsetM + row];
                            Bsub[row][col] = pB[tiledIndex*N + offsetN + row];
                        }
                        BARRIER();
                        for (int k=0; k<TSK; k++) {
                            #pragma unroll
                            for (int wn=0; wn<WPTN; wn++) {
                                int col = tidn + wn*RTSN;
                                Breg[wn] = Bsub[col][k];
                            }
                            #pragma unroll
                            for (int wm=0; wm<WPTM; wm++) {
                                int row = tidm + wm*RTSM;
                                float Areg = Asub[k][row];
                                #pragma unroll
                                for (int wn=0; wn<WPTN; wn++) {
                                    acc[wm][wn] += Areg * Breg[wn];
                                }
                            }
                        }
                        BARRIER();
                        t++;
                    } while (t<numTiles);
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        int globalRow = offsetM + tidm + wm*RTSM;
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) {
                            int globalCol = offsetN + tidn + wn*RTSN;
                            pC[globalCol*M + globalRow] = acc[wm][wn];
                        }
                    }
                });
            });

            // ============================== KERNEL 7 and 8 =====================================
            // (kernel 8 only differs from 7 by CUDA-specific __ldg / __shfl, which are not used)
            #elif KERNEL == 7 || KERNEL == 8
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(TSK, TSM), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(TSK, TSN), h);
                h.parallel_for(ndr(M/WPTM, N/WPTN, TSM/WPTM, TSN/WPTN), [=](sycl::nd_item<2> it) {
                    const floatX* A4 = reinterpret_cast<const floatX*>(pA);
                    const floatX* B4 = reinterpret_cast<const floatX*>(pB);
                    const int tidm = L0;
                    const int tidn = L1;
                    const int offsetM = TSM*G0;
                    const int offsetN = TSN*G1;
                    float Breg[WPTN];
                    float acc[WPTM][WPTN];
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) { acc[wm][wn] = 0.0f; }
                    }
                    const int numTiles = K/TSK;
                    int t=0;
                    do {
                        #pragma unroll
                        for (int la=0; la<LPTA/WIDTH; la++) {
                            int tid = tidn*RTSM + tidm;
                            int id = la*RTSN*RTSM + tid;
                            int row = MOD2(id,TSM/WIDTH);
                            int col = DIV2(id,TSM/WIDTH);
                            int tiledIndex = TSK*t + col;
                            floatX vecA = A4[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
                            floatX vecB = B4[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];
                            #pragma unroll
                            for (int i=0; i<WIDTH; i++) {
                                Asub[col][WIDTH*row + i] = vecA[i];
                                Bsub[col][WIDTH*row + i] = vecB[i];
                            }
                        }
                        BARRIER();
                        #pragma unroll
                        for (int k=0; k<TSK; k++) {
                            #pragma unroll
                            for (int wn=0; wn<WPTN; wn++) {
                                int col = tidn + wn*RTSN;
                                Breg[wn] = Bsub[k][col];
                            }
                            #pragma unroll
                            for (int wm=0; wm<WPTM; wm++) {
                                int row = tidm + wm*RTSM;
                                float Areg = Asub[k][row];
                                #pragma unroll
                                for (int wn=0; wn<WPTN; wn++) {
                                    acc[wm][wn] += Areg * Breg[wn];
                                }
                            }
                        }
                        BARRIER();
                        t++;
                    } while (t<numTiles);
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        int globalRow = offsetM + tidm + wm*RTSM;
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) {
                            int globalCol = offsetN + tidn + wn*RTSN;
                            pC[globalCol*M + globalRow] = acc[wm][wn];
                        }
                    }
                });
            });

            // ================================ KERNEL 9 =========================================
            #elif KERNEL == 9
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(2, TSK*TSM), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(2, TSK*TSN), h);
                h.parallel_for(ndr(M/WPTM, N/WPTN, TSM/WPTM, TSN/WPTN), [=](sycl::nd_item<2> it) {
                    const floatX* A4 = reinterpret_cast<const floatX*>(pA);
                    const floatX* B4 = reinterpret_cast<const floatX*>(pB);
                    const int tidm = L0;
                    const int tidn = L1;
                    const int offsetM = TSM*G0;
                    const int offsetN = TSN*G1;
                    float Breg[WPTN];
                    float acc[WPTM][WPTN];
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) { acc[wm][wn] = 0.0f; }
                    }

                    // Load the first tile of A and B into local memory
                    #pragma unroll
                    for (int la=0; la<LPTA/WIDTH; la++) {
                        int tid = tidn*RTSM + tidm;
                        int id = la*RTSN*RTSM + tid;
                        int row = MOD2(id,TSM/WIDTH);
                        int col = DIV2(id,TSM/WIDTH);
                        int tiledIndex = TSK*0 + col;
                        floatX vecA = A4[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
                        floatX vecB = B4[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];
                        #pragma unroll
                        for (int i=0; i<WIDTH; i++) {
                            Asub[0][col*TSM + WIDTH*row + i] = vecA[i];
                            Bsub[0][col*TSN + WIDTH*row + i] = vecB[i];
                        }
                    }
                    BARRIER();

                    const int numTiles = K/TSK;
                    int t=0;
                    do {
                        // Load the next tile of A and B into local memory
                        int tt = t + 1;
                        if (tt < numTiles) {
                            #pragma unroll
                            for (int la=0; la<LPTA/WIDTH; la++) {
                                int tid = tidn*RTSM + tidm;
                                int id = la*RTSN*RTSM + tid;
                                int row = MOD2(id,TSM/WIDTH);
                                int col = DIV2(id,TSM/WIDTH);
                                int tiledIndex = TSK*tt + col;
                                floatX vecA = A4[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
                                floatX vecB = B4[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];
                                #pragma unroll
                                for (int i=0; i<WIDTH; i++) {
                                    Asub[tt%2][col*TSM + WIDTH*row + i] = vecA[i];
                                    Bsub[tt%2][col*TSN + WIDTH*row + i] = vecB[i];
                                }
                            }
                        }
                        #pragma unroll
                        for (int k=0; k<TSK; k++) {
                            #pragma unroll
                            for (int wn=0; wn<WPTN; wn++) {
                                int col = tidn + wn*RTSN;
                                Breg[wn] = Bsub[t%2][k*TSN + col];
                            }
                            #pragma unroll
                            for (int wm=0; wm<WPTM; wm++) {
                                int row = tidm + wm*RTSM;
                                float Areg = Asub[t%2][k*TSM + row];
                                #pragma unroll
                                for (int wn=0; wn<WPTN; wn++) {
                                    acc[wm][wn] += Areg * Breg[wn];
                                }
                            }
                        }
                        BARRIER();
                        t++;
                    } while (t<numTiles);
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        int globalRow = offsetM + tidm + wm*RTSM;
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) {
                            int globalCol = offsetN + tidn + wn*RTSN;
                            pC[globalCol*M + globalRow] = acc[wm][wn];
                        }
                    }
                });
            });

            // ================================ KERNEL 10 ========================================
            #elif KERNEL == 10
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> Asub(sycl::range<2>(2, TSK*TSM), h);
                sycl::local_accessor<float, 2> Bsub(sycl::range<2>(2, TSK*TSN), h);
                h.parallel_for(ndr(M/WPTM, N/WPTN, TSM/WPTM, TSN/WPTN), [=](sycl::nd_item<2> it) {
                    const floatX* A4 = reinterpret_cast<const floatX*>(pA);
                    const floatX* B4 = reinterpret_cast<const floatX*>(pB);
                    const int tidm = L0;
                    const int tidn = L1;
                    const int gidm = G0;
                    const int gidn = G1;
                    const int tid = tidn*RTSM + tidm;
                    float Breg[WPTN];
                    float acc[WPTM][WPTN];
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) { acc[wm][wn] = 0.0f; }
                    }

                    // Tile A (first tile)
                    #pragma unroll
                    for (int la=0; la<LPTA/WIDTH; la++) {
                        int id = la*RTSN*RTSM + tid;
                        int row = MOD2(id,TSM/WIDTH);
                        int col = DIV2(id,TSM/WIDTH);
                        int tiledIndex = TSK*0 + col;
                        floatX vecA = A4[tiledIndex*(M/WIDTH) + gidm*(TSM/WIDTH) + row];
                        #pragma unroll
                        for (int i=0; i<WIDTH; i++) { Asub[0][col*TSM + WIDTH*row + i] = vecA[i]; }
                    }
                    // Tile B (first tile)
                    #pragma unroll
                    for (int lb=0; lb<LPTB/WIDTH; lb++) {
                        int id = lb*RTSN*RTSM + tid;
                        int row = MOD2(id,TSN/WIDTH);
                        int col = DIV2(id,TSN/WIDTH);
                        int tiledIndex = TSK*0 + col;
                        floatX vecB = B4[tiledIndex*(N/WIDTH) + gidn*(TSN/WIDTH) + row];
                        #pragma unroll
                        for (int i=0; i<WIDTH; i++) { Bsub[0][col*TSN + WIDTH*row + i] = vecB[i]; }
                    }

                    const int numTiles = K/TSK;
                    int t=0;
                    do {
                        BARRIER();

                        // Load the next tile of A and B into local memory
                        int tt = t + 1;
                        if (tt < numTiles) {
                            #pragma unroll
                            for (int la=0; la<LPTA/WIDTH; la++) {
                                int id = la*RTSN*RTSM + tid;
                                int row = MOD2(id,TSM/WIDTH);
                                int col = DIV2(id,TSM/WIDTH);
                                int tiledIndex = TSK*tt + col;
                                floatX vecA = A4[tiledIndex*(M/WIDTH) + gidm*(TSM/WIDTH) + row];
                                #pragma unroll
                                for (int i=0; i<WIDTH; i++) {
                                    Asub[tt%2][col*TSM + WIDTH*row + i] = vecA[i];
                                }
                            }
                            #pragma unroll
                            for (int lb=0; lb<LPTB/WIDTH; lb++) {
                                int id = lb*RTSN*RTSM + tid;
                                int row = MOD2(id,TSN/WIDTH);
                                int col = DIV2(id,TSN/WIDTH);
                                int tiledIndex = TSK*tt + col;
                                floatX vecB = B4[tiledIndex*(N/WIDTH) + gidn*(TSN/WIDTH) + row];
                                #pragma unroll
                                for (int i=0; i<WIDTH; i++) {
                                    Bsub[tt%2][col*TSN + WIDTH*row + i] = vecB[i];
                                }
                            }
                        }
                        #pragma unroll
                        for (int k=0; k<TSK; k++) {
                            #pragma unroll
                            for (int wn=0; wn<WPTN; wn++) {
                                int col = tidn + wn*RTSN;
                                Breg[wn] = Bsub[t%2][k*TSN + col];
                            }
                            #pragma unroll
                            for (int wm=0; wm<WPTM; wm++) {
                                int row = tidm + wm*RTSM;
                                float Areg = Asub[t%2][k*TSM + row];
                                #pragma unroll
                                for (int wn=0; wn<WPTN; wn++) {
                                    acc[wm][wn] += Areg * Breg[wn];
                                }
                            }
                        }
                        t++;
                    } while (t<numTiles);
                    #pragma unroll
                    for (int wm=0; wm<WPTM; wm++) {
                        int globalRow = gidm*TSM + tidm + wm*RTSM;
                        #pragma unroll
                        for (int wn=0; wn<WPTN; wn++) {
                            int globalCol = gidn*TSN + tidn + wn*RTSN;
                            pC[globalCol*M + globalRow] = acc[wm][wn];
                        }
                    }
                });
            });

            // ================================ KERNEL 11 ========================================
            #elif KERNEL == 11
            q.submit([&](sycl::handler& h) {
                h.parallel_for(ndr(M/RX, N/RY, THREADSX, THREADSY), [=](sycl::nd_item<2> it) {
                    typedef sycl::vec<float, RX> floatA;
                    typedef sycl::vec<float, RK> floatB;
                    const floatA* A4 = reinterpret_cast<const floatA*>(pA);
                    const floatB* B4 = reinterpret_cast<const floatB*>(pB);
                    floatA* C4 = reinterpret_cast<floatA*>(pC);
                    float aReg[RK][RX];
                    float bReg[RY][RK];
                    float acc[RY][RX];
                    #pragma unroll
                    for (int y=0; y<RY; y++) {
                        for (int x=0; x<RX; x++) { acc[y][x] = 0.0f; }
                    }
                    const int numTiles = K/RK;
                    for (int t=0; t<numTiles; t++) {
                        #pragma unroll
                        for (int y=0; y<RY; y++) {
                            floatA aVec = A4[(RK*t + y)*(M/RX) + GL0];
                            floatB bVec = B4[(RY*GL1 + y)*numTiles + t];
                            #pragma unroll
                            for (int x=0; x<RX; x++) { aReg[y][x] = aVec[x]; }
                            #pragma unroll
                            for (int k=0; k<RK; k++) { bReg[y][k] = bVec[k]; }
                        }
                        #pragma unroll
                        for (int k=0; k<RK; k++) {
                            #pragma unroll
                            for (int y=0; y<RY; y++) {
                                #pragma unroll
                                for (int x=0; x<RX; x++) {
                                    acc[y][x] += aReg[k][x] * bReg[y][k];
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int y=0; y<RY; y++) {
                        floatA accVec;
                        #pragma unroll
                        for (int x=0; x<RX; x++) { accVec[x] = acc[y][x]; }
                        C4[(y + RY*GL1) * (M/RX) + GL0] = accVec;
                    }
                });
            });
            #else
                #error "KERNEL must be 1 -- 11"
            #endif
        }

        // ---- Remove the padding from the larger output (myGEMM10 only) -------------------------
        #if KERNEL == 10
        {
            const float* in = dC_XL;
            float* out = dC;
            q.submit([&](sycl::handler& h) {
                h.parallel_for(ndr(up(M,PADDINGX), up(N,PADDINGY), PADDINGX, PADDINGY),
                               [=](sycl::nd_item<2> it) {
                    const int tx = G0*PADDINGX + L0;
                    const int ty = G1*PADDINGY + L1;
                    if (tx < M && ty < N) {
                        out[ty*M + tx] = in[ty*M_XL + tx];
                    }
                });
            });
        }
        #endif

        // Wait for calculations to be finished
        q.wait();
    };

    // Untimed warm-up run (triggers the JIT compilation of the kernels), then the timed loop
    run();
    double startTime = timer();
    for (int r=0; r<NUM_RUNS; r++) { run(); }
    timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    timers[timerID].kf += ((long long)K * (long long)M * (long long)N * 2) / 1000;

    // Copy the output matrix C back to the CPU memory
    q.memcpy(C, dC, (size_t)M*N*sizeof(float)).wait();

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dB_TR, q);
    sycl::free(dC, q);
    sycl::free(dA_XL, q);
    sycl::free(dB_TR_XL, q);
    sycl::free(dC_XL, q);
}

// =================================================================================================
