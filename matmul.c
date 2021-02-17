#include <arm_neon.h>
#include <stdio.h>
#include <papi.h>

typedef uint16_t u16;

#define N 8
#define MAXRAND 8

#if (N % 8) == 0

#define BLOCKSIZE 8
typedef uint16x8_t vec;
typedef uint16x8x4_t vecval;
#define vload(c, ptr) c = vld1q_u16(ptr);
#define vloadx4(c, ptr) c = vld1q_u16_x4(ptr);
#define vstore(ptr, c) vst1q_u16(ptr, c);
#define vstorex4(ptr, c) vst1q_u16_x4(ptr, c);
#define vmla(c, a, b) c = vmlaq_u16(c, a, b);
#define vmlalane(c, a, b, n) c = vmlaq_laneq_u16(c, a, b, n);
#define vdup(c, n) c = vdupq_n_u16(n);

#elif (N % 4) == 0

#define size 4
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

void classicMatMult(u16 A[], u16 B[], u16 C[])
{
	for (int i = 0; i < N; i++)
		for (int k = 0; k < N; k++)
			for (int j = 0; j < N; j++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
				// printf("%d * %d\n", i * N + k , k * N + j);
			}
}


void inplace_tranpose(u16 *dst, u16 *src)
{
	int S = 4;
	for (int i = 0; i < N; i += S)
	{
		for (int j = 0; j < N; j += S)
		{
			// transpose smaller block size SxS
			for (int m = i; m < i + S; ++m)
			{
				for (int n = j; n < j + S; ++n)
				{
					dst[m + n * N] = src[n + m * N];
				}
			}
		}
	}
}

void neonMatMul_ref(u16 AT[], u16 B[], u16 C[])
{
	vec va, vb, vc;
	int i = 0, j = 0, k = 0;
	int index_AT, index_B;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			for (k = 0; k < N; k++)
			{
				index_AT = j * N + k;
				index_B = i * N + k;
				C[i * N + j] += B[index_B] * AT[index_AT];
				printf("%d: %d * %d\n", i * N + j, index_AT, index_B);
			}
		}
	}
}

void neonMatMul_tranpose(u16 AT[], u16 B[], u16 C[])
{
	vec va, vb, vc;
	int i = 0, j = 0, k = 0;
	int index_AT, index_B;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			for (k = 0; k < N; k += BLOCKSIZE)
			{
				index_AT = j * N + k;
				index_B = i * N + k;
				C[i * N + j] += B[index_B] * AT[index_AT];
				printf("%d: %d * %d\n", i * N + j, index_AT, index_B);
			}
		}
	}
}

#if (N % 8) == 0
void neonMatMul_base(u16 A[], u16 B[], u16 C[])
{
	vecval va1, va2, vb1, vb2, vc;

	vloadx4(vb1, &B[0]);
	vloadx4(vb2, &B[32]);

	vloadx4(va1, &A[0]);
	vloadx4(va2, &A[32]);

	// vload(vb1.val[0], &B[0 * 8]);
	// vload(vb1.val[1], &B[1 * 8]);
	// vload(vb1.val[2], &B[2 * 8]);
	// vload(vb1.val[3], &B[3 * 8]);

	// vload(vb2.val[0], &B[4 * 8]);
	// vload(vb2.val[1], &B[5 * 8]);
	// vload(vb2.val[2], &B[6 * 8]);
	// vload(vb2.val[3], &B[7 * 8]);

	// vload(va1.val[0], &A[0 * 8]);
	// vload(va1.val[1], &A[1 * 8]);
	// vload(va1.val[2], &A[2 * 8]);
	// vload(va1.val[3], &A[3 * 8]);

	// vload(va2.val[0], &A[4 * 8]);
	// vload(va2.val[1], &A[5 * 8]);
	// vload(va2.val[2], &A[6 * 8]);
	// vload(va2.val[3], &A[7 * 8]);

	vloadx4(vc, &C[0]);

	for (int i = 0; i < 4; i++)
	{
		// vload(vc, &C[i * 8]); 

		// vdup(va1.val[0], AT[i * 8 + 0]);
		// vdup(va1.val[1], AT[i * 8 + 1]);
		// vdup(va1.val[2], AT[i * 8 + 2]);
		// vdup(va1.val[3], AT[i * 8 + 3]);

		// vdup(va1.val[0], AT[i * 8 + 4]);
		// vdup(va1.val[1], AT[i * 8 + 5]);
		// vdup(va1.val[2], AT[i * 8 + 6]);
		// vdup(va1.val[3], AT[i * 8 + 7]);
		
		// vmla(vc, va1.val[0], vb1.val[0]);
		// vmla(vc, va1.val[1], vb1.val[1]);
		// vmla(vc, va1.val[2], vb1.val[2]);
		// vmla(vc, va1.val[3], vb1.val[3]);

		// vmla(vc, va2.val[0], vb2.val[0]);
		// vmla(vc, va2.val[1], vb2.val[1]);
		// vmla(vc, va2.val[2], vb2.val[2]);
		// vmla(vc, va2.val[3], vb2.val[3]);

		vmlalane(vc.val[i], va1.val[0], vb1.val[i], 0);
		vmlalane(vc.val[i], va1.val[1], vb1.val[i], 1);
		vmlalane(vc.val[i], va1.val[2], vb1.val[i], 2);
		vmlalane(vc.val[i], va1.val[3], vb1.val[i], 3);

		vmlalane(vc.val[i], va2.val[0], vb1.val[i], 4);
		vmlalane(vc.val[i], va2.val[1], vb1.val[i], 5);
		vmlalane(vc.val[i], va2.val[2], vb1.val[i], 6);
		vmlalane(vc.val[i], va2.val[3], vb1.val[i], 7);

		// vstore(&C[i * 8], vc);
	}
	vstorex4(&C[0], vc);

	vloadx4(vc, &C[32]);

	for (int i = 4; i < 8; i++)
	{
		// vload(vc, &C[i * 8]); 

		// vdup(va1.val[0], AT[i * 8 + 0]);
		// vdup(va1.val[1], AT[i * 8 + 1]);
		// vdup(va1.val[2], AT[i * 8 + 2]);
		// vdup(va1.val[3], AT[i * 8 + 3]);

		// vdup(va1.val[0], AT[i * 8 + 4]);
		// vdup(va1.val[1], AT[i * 8 + 5]);
		// vdup(va1.val[2], AT[i * 8 + 6]);
		// vdup(va1.val[3], AT[i * 8 + 7]);
		
		// vmla(vc, va1.val[0], vb1.val[0]);
		// vmla(vc, va1.val[1], vb1.val[1]);
		// vmla(vc, va1.val[2], vb1.val[2]);
		// vmla(vc, va1.val[3], vb1.val[3]);

		// vmla(vc, va2.val[0], vb2.val[0]);
		// vmla(vc, va2.val[1], vb2.val[1]);
		// vmla(vc, va2.val[2], vb2.val[2]);
		// vmla(vc, va2.val[3], vb2.val[3]);

		vmlalane(vc.val[i-4], va1.val[0], vb2.val[i-4], 0);
		vmlalane(vc.val[i-4], va1.val[1], vb2.val[i-4], 1);
		vmlalane(vc.val[i-4], va1.val[2], vb2.val[i-4], 2);
		vmlalane(vc.val[i-4], va1.val[3], vb2.val[i-4], 3);

		vmlalane(vc.val[i-4], va2.val[0], vb2.val[i-4], 4);
		vmlalane(vc.val[i-4], va2.val[1], vb2.val[i-4], 5);
		vmlalane(vc.val[i-4], va2.val[2], vb2.val[i-4], 6);
		vmlalane(vc.val[i-4], va2.val[3], vb2.val[i-4], 7);

	}
	vstorex4(&C[32], vc);


}

#elif (N % 4) == 0

void neonMatMul_base(u16 A[], u16 B[], u16 C[])
{
	vecval va, vb;
	vec vc;

	vload(vb.val[0], &B[0 * N]);
	vload(vb.val[1], &B[1 * N]);
	vload(vb.val[2], &B[2 * N]);
	vload(vb.val[3], &B[3 * N]);

	vload(va.val[0], &A[0 * N]);
	vload(va.val[1], &A[1 * N]);
	vload(va.val[2], &A[2 * N]);
	vload(va.val[3], &A[3 * N]);
	for (int i = 0; i < N; i++)
	{
		// vload(vc, &C[i * N]); 
		vdup(vc, 0);

		// vdup(va.val[0], AT[i * 4 + 0]);
		// vdup(va.val[1], AT[i * 4 + 1]);
		// vdup(va.val[2], AT[i * 4 + 2]);
		// vdup(va.val[3], AT[i * 4 + 3]);
		
		// vmla(vc, va.val[0], vb.val[0]);
		// vmla(vc, va.val[1], vb.val[1]);
		// vmla(vc, va.val[2], vb.val[2]);
		// vmla(vc, va.val[3], vb.val[3]);

		vmlalane(vc, va.val[0], vb.val[i], 0);
		vmlalane(vc, va.val[1], vb.val[i], 1);
		vmlalane(vc, va.val[2], vb.val[i], 2);
		vmlalane(vc, va.val[3], vb.val[i], 3);

		vstore(&C[i * N], vc);
	}
}
#else
void neonMatMul_base(u16 AT[], u16 B[], u16 C[])
{
	// for (int i = 0; i < N; i++)
	// {
	// 	for (int k = 0; k < N; k++)
	// 	{
	// 		for (int j =0; j < N; j++)
	// 		{
	// 			C[i*N + j] += AT[i*N + k] * B[k*N + j];
	// 		}
	// 	}
	// }

	/* vec vc, va, vb;
	for (int i = 0; i < N; i++)
	{
		for (int k = 0; k < N; k++)
		{
			vdup(va, AT[i*N + k]);
			for (int j =0; j < N; j+=size)
			{
				vload(vc, &C[i*N + j]);
				vload(vb, &B[k*N + j]);

				// C[i*N + j] += AT[i*N + k] * B[k*N + j];
				vmla(vc, va, vb);

				vstore(&C[i*N+j], vc);
			}
		}
	} */
	vec vc, va, vb;
	for (int i = 0; i < N; i++)
	{
		for (int k = 0; k < N; k++)
		{
			vdup(va, AT[i*N + k]);
			vload(vc, &C[i*N]);
			vload(vb, &B[k*N]);

			vmla(vc, va, vb);

			vstore(&C[i*N], vc);
			
		}
	}
}

#endif

int checkCorrect(u16 A[], u16 B[])
{
	int result = 0;
	for (int i = 0; i < N * N; i++)
	{
		if (A[i] != B[i])
		{
			// printf("%d: %d != %d\n", i, A[i], B[i]);
			result = 1;
		}
	}
	return result;
}

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

#define TESTS 1

int main()
{
	u16 A[N * N], B[N * N], C[N * N] = {0}, D[N * N] = {0};
	u16 M[N * N];
	for (int i = 0; i < N * N; i++)
	{
		// A[i] = (u16)rand() % MAXRAND;
		A[i] = i;
		// B[i] = (u16)rand() % MAXRAND;
		B[i] = i;
	}

	// print_array(A);
	// print_array(B);

	// PAPI_hl_region_begin("classic");
	for (int i = 0; i < TESTS; i++)
	{
		classicMatMult(A, B, C);
	}
	// PAPI_hl_region_end("classic");

	// inplace_tranpose(M, A);
	// print_array(M);
	// PAPI_hl_region_begin("neon");
	for (int i = 0; i < TESTS; i++)
	{
		neonMatMul_base(A, B, D);
	}
	// PAPI_hl_region_end("neon");

	print_array(C);
	print_array(D);

	if (checkCorrect(C, D))
	{
		printf("Error\n");
		return 1;
	}
	return 0;
}
