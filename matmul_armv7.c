#include <arm_neon.h>
#include <stdio.h>

typedef uint16_t u16;

// Correct with N = 4, 12, 20
// gcc matmul.c -o matmul -O3  -g3 -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon


#define N 8
#define MAXRAND 8

#if (N % 8) == 0

#define BLOCKSIZE 8
#define BLOCKSIZE_HALF 4
typedef uint16x8_t vec;
typedef uint16x8x4_t vecval;
typedef uint16x4x4_t vecval2;
#define vload(c, ptr) c = vld1q_u16(ptr);
#define vload_half(c, ptr) c = vld1_u16(ptr);
#define vloadx4(c, ptr) c = vld1q_u16_x4(ptr);
#define vstore(ptr, c) vst1q_u16(ptr, c);
#define vstorex4(ptr, c) vst1q_u16_x4(ptr, c);
#define vmla(c, a, b) c = vmlaq_u16(c, a, b);
#define vmlalane(c, a, b, n) c = vmlaq_lane_u16(c, a, b, n);
#define vdup(c, n) c = vdupq_n_u16(n);

#elif (N % 4) == 0

#define BLOCKSIZE 4
typedef uint16x4_t vec;
typedef uint16x4x4_t vecval;
#define vload(c, ptr) c = vld1_u16(ptr);
#define vloadx4(c, ptr) c = vld1_u16_x4(ptr);
#define vstore(ptr, c) vst1_u16(ptr, c);
#define vstorex4(ptr, c) vst1_u16_x4(ptr, c);
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

void print_vector_half(vecval2 a)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < BLOCKSIZE_HALF; j++)
		{
			printf("%4d, ", a.val[i][j]);
		}
		printf("\n");
	}
	printf("--\n");
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
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
				// printf("%d * %d\n", i * N + k , k * N + j);
			}
}

#if (N % 8) == 0
void neonMatMul_base(u16 A[], u16 B[], u16 C[])
{
	// Total registers:  24
	vecval va1, va2, vc1, vc2;
    vecval2 vb1, vb2, vb3, vb4;

	vload_half(vb1.val[0], &B[0 * N]);
	vload_half(vb1.val[1], &B[0 * N + 4]);
	vload_half(vb1.val[2], &B[1 * N]);
	vload_half(vb1.val[3], &B[1 * N + 4]);

	vload_half(vb2.val[0], &B[2 * N]);
	vload_half(vb2.val[1], &B[2 * N + 4]);
	vload_half(vb2.val[2], &B[3 * N]);
	vload_half(vb2.val[3], &B[3 * N + 4]);

    vload_half(vb3.val[0], &B[4 * N]);
	vload_half(vb3.val[1], &B[4 * N + 4]);
	vload_half(vb3.val[2], &B[5 * N]);
	vload_half(vb3.val[3], &B[5 * N + 4]);

	vload_half(vb4.val[0], &B[6 * N]);
	vload_half(vb4.val[1], &B[6 * N + 4]);
	vload_half(vb4.val[2], &B[7 * N]);
	vload_half(vb4.val[3], &B[7 * N + 4]);

    print_vector_half(vb1);
    print_vector_half(vb2);
    print_vector_half(vb3);
    print_vector_half(vb4);

	vload(va1.val[0], &A[0 * N]);
	vload(va1.val[1], &A[1 * N]);
	vload(va1.val[2], &A[2 * N]);
	vload(va1.val[3], &A[3 * N]);

	vload(va2.val[0], &A[4 * N]);
	vload(va2.val[1], &A[5 * N]);
	vload(va2.val[2], &A[6 * N]);
	vload(va2.val[3], &A[7 * N]);

	vload(vc1.val[0], &C[0 * N]);
	vload(vc1.val[1], &C[1 * N]);
	vload(vc1.val[2], &C[2 * N]);
	vload(vc1.val[3], &C[3 * N]);

	vload(vc2.val[0], &C[4 * N]);
	vload(vc2.val[1], &C[5 * N]);
	vload(vc2.val[2], &C[6 * N]);
	vload(vc2.val[3], &C[7 * N]);

    print_vector(va1);
    print_vector(va2);
    print_vector(vc1);
    print_vector(vc2);


    vmlalane(vc1.val[0], va1.val[0],  vb1.val[0], 0);
    // print_vector(vc1); printf("%d\n", vb1.val[0][0]);
    vmlalane(vc1.val[0], va1.val[1],  vb1.val[0], 1);
    // print_vector(vc1); printf("%d\n", vb1.val[0][1]);
    vmlalane(vc1.val[0], va1.val[2],  vb1.val[0], 2);
    // print_vector(vc1); printf("%d\n", vb1.val[0][2]);
    vmlalane(vc1.val[0], va1.val[3],  vb1.val[0], 3);
    // print_vector(vc1); printf("%d\n", vb1.val[0][3]);

    vmlalane(vc1.val[0], va2.val[0],  vb1.val[0+1], 0);
    // print_vector(vc1); printf("%d\n", vb1.val[0+1][0]);
    vmlalane(vc1.val[0], va2.val[1],  vb1.val[0+1], 1);
    // print_vector(vc1); printf("%d\n", vb1.val[0+1][1]);
    vmlalane(vc1.val[0], va2.val[2],  vb1.val[0+1], 2);
    // print_vector(vc1); printf("%d\n", vb1.val[0+1][2]);
    vmlalane(vc1.val[0], va2.val[3],  vb1.val[0+1], 3);
    // print_vector(vc1); printf("%d\n", vb1.val[0+1][3]);

    vmlalane(vc1.val[1], va1.val[0],  vb1.val[0+2], 0);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+2][0]);
    vmlalane(vc1.val[1], va1.val[1],  vb1.val[0+2], 1);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+2][1]);
    vmlalane(vc1.val[1], va1.val[2],  vb1.val[0+2], 2);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+2][2]);
    vmlalane(vc1.val[1], va1.val[3],  vb1.val[0+2], 3);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+2][3]);

    vmlalane(vc1.val[1], va2.val[0],  vb1.val[0+3], 0);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+3][0]);
    vmlalane(vc1.val[1], va2.val[1],  vb1.val[0+3], 1);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+3][1]);
    vmlalane(vc1.val[1], va2.val[2],  vb1.val[0+3], 2);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+3][2]);
    vmlalane(vc1.val[1], va2.val[3],  vb1.val[0+3], 3);
    // print_vector(vc1);   printf("%d\n", vb1.val[0+3][3]);

    vmlalane(vc1.val[2], va1.val[0],  vb2.val[0], 0);
    // print_vector(vc1); printf("%d\n", vb2.val[0][0]);
    vmlalane(vc1.val[2], va1.val[1],  vb2.val[0], 1);
    // print_vector(vc1); printf("%d\n", vb2.val[0][1]);
    vmlalane(vc1.val[2], va1.val[2],  vb2.val[0], 2);
    // print_vector(vc1); printf("%d\n", vb2.val[0][2]);
    vmlalane(vc1.val[2], va1.val[3],  vb2.val[0], 3);
    // print_vector(vc1); printf("%d\n", vb2.val[0][3]);

    vmlalane(vc1.val[2], va2.val[0],  vb2.val[0+1], 0);
    // print_vector(vc1); printf("%d\n", vb2.val[0+1][0]);
    vmlalane(vc1.val[2], va2.val[1],  vb2.val[0+1], 1);
    // print_vector(vc1); printf("%d\n", vb2.val[0+1][1]);
    vmlalane(vc1.val[2], va2.val[2],  vb2.val[0+1], 2);
    // print_vector(vc1); printf("%d\n", vb2.val[0+1][2]);
    vmlalane(vc1.val[2], va2.val[3],  vb2.val[0+1], 3);
    // print_vector(vc1); printf("%d\n", vb2.val[0+1][3]);

    vmlalane(vc1.val[3], va1.val[0],  vb2.val[0+2], 0);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+2][0]);
    vmlalane(vc1.val[3], va1.val[1],  vb2.val[0+2], 1);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+2][1]);
    vmlalane(vc1.val[3], va1.val[2],  vb2.val[0+2], 2);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+2][2]);
    vmlalane(vc1.val[3], va1.val[3],  vb2.val[0+2], 3);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+2][3]);

    vmlalane(vc1.val[3], va2.val[0],  vb2.val[0+3], 0);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+3][0]);
    vmlalane(vc1.val[3], va2.val[1],  vb2.val[0+3], 1);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+3][1]);
    vmlalane(vc1.val[3], va2.val[2],  vb2.val[0+3], 2);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+3][2]);
    vmlalane(vc1.val[3], va2.val[3],  vb2.val[0+3], 3);
    // print_vector(vc1);   printf("%d\n", vb2.val[0+3][3]);
	

	vmlalane(vc2.val[0], va1.val[0],  vb3.val[0], 0);
    // print_vector(vc2); printf("%d\n", vb3.val[0][0]);
    vmlalane(vc2.val[0], va1.val[1],  vb3.val[0], 1);
    // print_vector(vc2); printf("%d\n", vb3.val[0][1]);
    vmlalane(vc2.val[0], va1.val[2],  vb3.val[0], 2);
    // print_vector(vc2); printf("%d\n", vb3.val[0][2]);
    vmlalane(vc2.val[0], va1.val[3],  vb3.val[0], 3);
    // print_vector(vc2); printf("%d\n", vb3.val[0][3]);

    vmlalane(vc2.val[0], va2.val[0],  vb3.val[0+1], 0);
    // print_vector(vc2); printf("%d\n", vb3.val[0+1][0]);
    vmlalane(vc2.val[0], va2.val[1],  vb3.val[0+1], 1);
    // print_vector(vc2); printf("%d\n", vb3.val[0+1][1]);
    vmlalane(vc2.val[0], va2.val[2],  vb3.val[0+1], 2);
    // print_vector(vc2); printf("%d\n", vb3.val[0+1][2]);
    vmlalane(vc2.val[0], va2.val[3],  vb3.val[0+1], 3);
    // print_vector(vc2); printf("%d\n", vb3.val[0+1][3]);

    vmlalane(vc2.val[1], va1.val[0],  vb3.val[0+2], 0);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+2][0]);
    vmlalane(vc2.val[1], va1.val[1],  vb3.val[0+2], 1);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+2][1]);
    vmlalane(vc2.val[1], va1.val[2],  vb3.val[0+2], 2);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+2][2]);
    vmlalane(vc2.val[1], va1.val[3],  vb3.val[0+2], 3);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+2][3]);

    vmlalane(vc2.val[1], va2.val[0],  vb3.val[0+3], 0);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+3][0]);
    vmlalane(vc2.val[1], va2.val[1],  vb3.val[0+3], 1);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+3][1]);
    vmlalane(vc2.val[1], va2.val[2],  vb3.val[0+3], 2);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+3][2]);
    vmlalane(vc2.val[1], va2.val[3],  vb3.val[0+3], 3);
    // print_vector(vc2);   printf("%d\n", vb3.val[0+3][3]);

    vmlalane(vc2.val[2], va1.val[0],  vb4.val[0], 0);
    // print_vector(vc2); printf("%d\n", vb4.val[0][0]);
    vmlalane(vc2.val[2], va1.val[1],  vb4.val[0], 1);
    // print_vector(vc2); printf("%d\n", vb4.val[0][1]);
    vmlalane(vc2.val[2], va1.val[2],  vb4.val[0], 2);
    // print_vector(vc2); printf("%d\n", vb4.val[0][2]);
    vmlalane(vc2.val[2], va1.val[3],  vb4.val[0], 3);
    // print_vector(vc2); printf("%d\n", vb4.val[0][3]);

    vmlalane(vc2.val[2], va2.val[0],  vb4.val[0+1], 0);
    // print_vector(vc2); printf("%d\n", vb4.val[0+1][0]);
    vmlalane(vc2.val[2], va2.val[1],  vb4.val[0+1], 1);
    // print_vector(vc2); printf("%d\n", vb4.val[0+1][1]);
    vmlalane(vc2.val[2], va2.val[2],  vb4.val[0+1], 2);
    // print_vector(vc2); printf("%d\n", vb4.val[0+1][2]);
    vmlalane(vc2.val[2], va2.val[3],  vb4.val[0+1], 3);
    // print_vector(vc2); printf("%d\n", vb4.val[0+1][3]);

    vmlalane(vc2.val[3], va1.val[0],  vb4.val[0+2], 0);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+2][0]);
    vmlalane(vc2.val[3], va1.val[1],  vb4.val[0+2], 1);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+2][1]);
    vmlalane(vc2.val[3], va1.val[2],  vb4.val[0+2], 2);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+2][2]);
    vmlalane(vc2.val[3], va1.val[3],  vb4.val[0+2], 3);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+2][3]);

    vmlalane(vc2.val[3], va2.val[0],  vb4.val[0+3], 0);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+3][0]);
    vmlalane(vc2.val[3], va2.val[1],  vb4.val[0+3], 1);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+3][1]);
    vmlalane(vc2.val[3], va2.val[2],  vb4.val[0+3], 2);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+3][2]);
    vmlalane(vc2.val[3], va2.val[3],  vb4.val[0+3], 3);
    // print_vector(vc2);   printf("%d\n", vb4.val[0+3][3]);

    // print_vector(vc1);
    print_vector(vc2);
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

	print_vector(va);
	print_vector(vb);
	// printf("===+++=====\n");
	print_vector(vc);

	for (int i = 0; i < BLOCKSIZE; i++)
	{
		// vload(vc, &C[i * N]);
		// vdup(vc, 0);

		// vdup(va.val[0], AT[i * 4 + 0]);
		// vdup(va.val[1], AT[i * 4 + 1]);
		// vdup(va.val[2], AT[i * 4 + 2]);
		// vdup(va.val[3], AT[i * 4 + 3]);

		// vmla(vc, va.val[0], vb.val[0]);
		// vmla(vc, va.val[1], vb.val[1]);
		// vmla(vc, va.val[2], vb.val[2]);
		// vmla(vc, va.val[3], vb.val[3]);

		vmlalane(vc.val[i], va.val[0], vb.val[i], 0);
		vmlalane(vc.val[i], va.val[1], vb.val[i], 1);
		vmlalane(vc.val[i], va.val[2], vb.val[i], 2);
		vmlalane(vc.val[i], va.val[3], vb.val[i], 3);
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
		for (int j = 0; j < N; j += BLOCKSIZE)
		{
			for (int k = 0; k < N; k += BLOCKSIZE)
			{
				neonMatMul_base(&A[i + N*k], &B[N*j + k], &C[N*j + i]);
			}
		}
}

int checkCorrect(u16 A[], u16 B[])
{
	int result = 0;
	for (int i = 0; i < N * N; i++)
	{
		if (A[i] != B[i])
		{
			return 1;
		}
	}
	return 0;
}

#define TESTS 1

int main()
{
	u16 A[N * N], B[N * N], C[N * N] = {0}, D[N * N] = {0};
	u16 M[N * N];
	for (int i = 0; i < N * N; i++)
	{
		A[i] = (u16)rand() % MAXRAND;
		// A[i] = i ;//% MAXRAND;
		B[i] = (u16)rand() % MAXRAND;
		// B[i] = i ;//% MAXRAND;
		// C[i] = i ;
		// D[i] = i;
	}


	for (int i = 0; i < TESTS; i++)
	{
		classicMatMult(A, B, C);
	}


	for (int i = 0; i < TESTS; i++)
	{
		neoMatMul(A, B, D);
	}

	print_array(A);
	print_array(B);
	print_array(C);
	print_array(D);

	if (checkCorrect(C, D))
	{
		printf("Error\n");
		return 1;
	}
	printf("OK\n");
	return 0;
}