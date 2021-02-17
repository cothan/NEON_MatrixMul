#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <papi.h>

typedef uint16_t u16;

// Correct with N = 4, 12, 20
// gcc matmul.c -o matmul -O3  -g3 -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon

#define N 260
#define MAXRAND 8
#ifndef DEBUG
#define DEBUG 0
#endif

#if (N % 8) == 0

#define BLOCKSIZE 8
typedef uint16x8_t vec;
typedef uint16x8x4_t vecval;
#define vload(c, ptr) c = vld1q_u16(ptr);
#define vstore(ptr, c) vst1q_u16(ptr, c);
#define vmla(c, a, b) c = vmlaq_u16(c, a, b);
#define vmlalane(c, a, b, n) c = vmlaq_laneq_u16(c, a, b, n);
#define vdup(c, n) c = vdupq_n_u16(n);

#elif (N % 4) == 0

#define BLOCKSIZE 4
typedef uint16x4_t vec;
typedef uint16x4x4_t vecval;
#define vload(c, ptr) c = vld1_u16(ptr);
#define vstore(ptr, c) vst1_u16(ptr, c);
#define vmla(c, a, b) c = vmla_u16(c, a, b);
#define vmlalane(c, a, b, n) c = vmla_lane_u16(c, a, b, n);
#define vdup(c, n) c = vdup_n_u16(n);

#else
#error "Matrix must be multiple of {4, 8}"

#endif

void print_array(u16 *a)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%4d, ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("=======\n");
}

void print_vector(vecval a)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            printf("%4d, ", a.val[i][j]);
        }
        printf("\n");
    }
    printf("--\n");
}

void classicMatMult(u16 A[], u16 B[], u16 C[])
{
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
                // printf("%2d * %2d, ", A[i * N + k], B[k * N + j]);
            }
            // printf("\n");
        }
        // printf("\n");
    }
}

#if (N % 8) == 0
void neonMatMul_base(u16 A[], u16 B[], u16 C[])
{
    // Total registers:  24
    vecval va1, va2, vb1, vb2, vc1, vc2;

    vload(va1.val[0], &A[0 * N]);
    vload(va1.val[1], &A[1 * N]);
    vload(va1.val[2], &A[2 * N]);
    vload(va1.val[3], &A[3 * N]);

    vload(va2.val[0], &A[4 * N]);
    vload(va2.val[1], &A[5 * N]);
    vload(va2.val[2], &A[6 * N]);
    vload(va2.val[3], &A[7 * N]);

    vload(vb1.val[0], &B[0 * N]);
    vload(vb1.val[1], &B[1 * N]);
    vload(vb1.val[2], &B[2 * N]);
    vload(vb1.val[3], &B[3 * N]);

    vload(vb2.val[0], &B[4 * N]);
    vload(vb2.val[1], &B[5 * N]);
    vload(vb2.val[2], &B[6 * N]);
    vload(vb2.val[3], &B[7 * N]);

    vload(vc1.val[0], &C[0 * N]);
    vload(vc1.val[1], &C[1 * N]);
    vload(vc1.val[2], &C[2 * N]);
    vload(vc1.val[3], &C[3 * N]);

    vload(vc2.val[0], &C[4 * N]);
    vload(vc2.val[1], &C[5 * N]);
    vload(vc2.val[2], &C[6 * N]);
    vload(vc2.val[3], &C[7 * N]);

    for (int i = 0; i < 4; i++)
    {
        vmlalane(vc1.val[i], vb1.val[0], va1.val[i], 0);
        vmlalane(vc1.val[i], vb1.val[1], va1.val[i], 1);
        vmlalane(vc1.val[i], vb1.val[2], va1.val[i], 2);
        vmlalane(vc1.val[i], vb1.val[3], va1.val[i], 3);

        vmlalane(vc1.val[i], vb2.val[0], va1.val[i], 4);
        vmlalane(vc1.val[i], vb2.val[1], va1.val[i], 5);
        vmlalane(vc1.val[i], vb2.val[2], va1.val[i], 6);
        vmlalane(vc1.val[i], vb2.val[3], va1.val[i], 7);
    }

    for (int i = 4; i < 8; i++)
    {
        vmlalane(vc2.val[i - 4], vb1.val[0], va2.val[i - 4], 0);
        vmlalane(vc2.val[i - 4], vb1.val[1], va2.val[i - 4], 1);
        vmlalane(vc2.val[i - 4], vb1.val[2], va2.val[i - 4], 2);
        vmlalane(vc2.val[i - 4], vb1.val[3], va2.val[i - 4], 3);

        vmlalane(vc2.val[i - 4], vb2.val[0], va2.val[i - 4], 4);
        vmlalane(vc2.val[i - 4], vb2.val[1], va2.val[i - 4], 5);
        vmlalane(vc2.val[i - 4], vb2.val[2], va2.val[i - 4], 6);
        vmlalane(vc2.val[i - 4], vb2.val[3], va2.val[i - 4], 7);
    }
    vstore(&C[0 * N], vc1.val[0]);
    vstore(&C[1 * N], vc1.val[1]);
    vstore(&C[2 * N], vc1.val[2]);
    vstore(&C[3 * N], vc1.val[3]);

    vstore(&C[4 * N], vc2.val[0]);
    vstore(&C[5 * N], vc2.val[1]);
    vstore(&C[6 * N], vc2.val[2]);
    vstore(&C[7 * N], vc2.val[3]);
}

#elif (N % 4) == 0

void neonMatMul_base(u16 A[], u16 B[], u16 C[])
{
    vecval va, vb, vc;

    vload(vb.val[0], &B[0 * N]);
    vload(vb.val[1], &B[1 * N]);
    vload(vb.val[2], &B[2 * N]);
    vload(vb.val[3], &B[3 * N]);

    vload(va.val[0], &A[0 * N]);
    vload(va.val[1], &A[1 * N]);
    vload(va.val[2], &A[2 * N]);
    vload(va.val[3], &A[3 * N]);

    vload(vc.val[0], &C[0 * N]);
    vload(vc.val[1], &C[1 * N]);
    vload(vc.val[2], &C[2 * N]);
    vload(vc.val[3], &C[3 * N]);

    for (int i = 0; i < BLOCKSIZE; i++)
    {
        vmlalane(vc.val[i], vb.val[0], va.val[i], 0);
        vmlalane(vc.val[i], vb.val[1], va.val[i], 1);
        vmlalane(vc.val[i], vb.val[2], va.val[i], 2);
        vmlalane(vc.val[i], vb.val[3], va.val[i], 3);
    }
    vstore(&C[0 * N], vc.val[0]);
    vstore(&C[1 * N], vc.val[1]);
    vstore(&C[2 * N], vc.val[2]);
    vstore(&C[3 * N], vc.val[3]);
}
#else
#error "Matrix must be multiple of {4, 8}"
#endif

void neoMatMul(u16 A[], u16 B[], u16 C[])
{
    for (int i = 0; i < N; i += BLOCKSIZE)
    {
        for (int k = 0; k < N; k += BLOCKSIZE)
        {
            for (int j = 0; j < N; j += BLOCKSIZE)
            {
                neonMatMul_base(&A[i * N + k], &B[k * N + j], &C[N * i + j]);
            }
        }
    }
}

int checkCorrect(u16 A[], u16 B[])
{
    for (int i = 0; i < N * N; i++)
    {
        if (A[i] != B[i])
        {
            return 1;
        }
    }
    return 0;
}

#define TESTS 1000

int main()
{
    srand(time(0));
    u16 A[N * N], B[N * N], C[N * N] = {0}, D[N * N] = {0};
    for (int i = 0; i < N * N; i++)
    {
        A[i] = rand(); //% MAXRAND;
        B[i] = rand(); //% MAXRAND;
        // A[i] = (i + 4);
        // B[i] = i;
        // C[i] = i ;
        // D[i] = i;
    }

#if DEBUG
    printf("A======\n");
    print_array(A);
    printf("B======\n");
    print_array(B);
#endif

    PAPI_hl_region_begin("classicMatMult");
    for (int i = 0; i < TESTS; i++)
    {
        classicMatMult(A, B, C);
    }
    PAPI_hl_region_end("classicMatMult");

    PAPI_hl_region_begin("neoMatMul");
    for (int i = 0; i < TESTS; i++)
    {
        neoMatMul(A, B, D);
    }
    PAPI_hl_region_end("neoMatMul");

#if DEBUG
    printf("C======\n");
    print_array(C);
    printf("D======\n");
    print_array(D);
#endif

    if (checkCorrect(C, D))
    {
        printf("Error\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
