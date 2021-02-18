
/***************************** Include Files *********************************/

#include "xparameters.h"
#include "xtmrctr.h"
#include "xil_printf.h"
#include <arm_neon.h>
#include <stdlib.h>

/************************** Constant Definitions *****************************/

/*
 * The following constants map to the XPAR parameters created in the
 * xparameters.h file. They are only defined here such that a user can easily
 * change all the needed parameters in one place.
 */
#define TMRCTR_DEVICE_ID	XPAR_TMRCTR_0_DEVICE_ID


/*
 * This example only uses the 1st of the 2 timer counters contained in a
 * single timer counter hardware device
 */
#define TIMER_COUNTER_0	 0
#define MAXRAND 8 // maximum size of a random number in a u16 matrix element
#define N 4 // size of matrix will be N x N, where N in {4,8,12,16,20,24}

/**************************** Type Definitions *******************************/


/***************** Macros (Inline Functions) Definitions *********************/


/************************** Function Prototypes ******************************/

void classicMatMult(u16 [], u16 [], u16 []);
void neonMatMult(u16 [], u16 [], u16 []);
int checkCorrect(u16 [], u16 []);

/************************** Variable Definitions *****************************/

XTmrCtr TimerCounter; /* The instance of the Tmrctr Device */
u16 A[N*N], B[N*N], C[N*N], D[N*N] = {0};

int main(void)
{

/* Generate pseudorandom arrays */

	xil_printf("Matrix size: %d x %d\n\r",N,N);
	for (int i = 0; i < N*N; i++){
		A[i] = (u16)rand()%MAXRAND;
		B[i] = (u16)rand()%MAXRAND;
	}

	int Status;

	/*
	 * Initialize the timer counter so that it's ready to use,
	 * specify the device ID that is generated in xparameters.h
	 */
	Status = XTmrCtr_Initialize(&TimerCounter, TMRCTR_DEVICE_ID);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Perform a self-test to ensure that the hardware was built
	 * correctly, use the 1st timer in the device (0)
	 */
	Status = XTmrCtr_SelfTest(&TimerCounter, TIMER_COUNTER_0);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Enable the Autoreload mode of the timer counters.
	 */
	XTmrCtr_SetOptions(&TimerCounter, TIMER_COUNTER_0,
				XTC_AUTO_RELOAD_OPTION);

	/*
	 * Get a snapshot of the timer counter value before it's started
	 * to compare against later
	 */
	u32 Value1 = XTmrCtr_GetValue(&TimerCounter, TIMER_COUNTER_0);

	xil_printf("Starting timer value: %d\n\r",Value1);

	/* Time critical region for classic matrix multiplication */
	XTmrCtr_Start(&TimerCounter, TIMER_COUNTER_0);

	classicMatMult(A, B, C);

	XTmrCtr_Stop(&TimerCounter, TIMER_COUNTER_0);
	/* End time critical region for classic matrix multiplication */

	u32 Value2 = XTmrCtr_GetValue(&TimerCounter, TIMER_COUNTER_0);
	xil_printf("Classic Multiplication Timer: %d \n\r",Value2);

    if (N<=8){
    	xil_printf("Classic Multiplication Results:\n\r");
    	//output
    	for (int i = 0; i < N; i++){
    		for (int j = 0; j < N; j++)
    			xil_printf("%d \t",C[i*N+j]);
    		xil_printf("\n\r");
    	}
    }

    XTmrCtr_Reset(&TimerCounter, TIMER_COUNTER_0);
	/* Time critical region for Neon matrix multiplication */

    XTmrCtr_Start(&TimerCounter, TIMER_COUNTER_0);

    neonMatMult(A, B, D);

    XTmrCtr_Stop(&TimerCounter, TIMER_COUNTER_0);
	/* End time critical region for Neon matrix multiplication */

    Value2 = XTmrCtr_GetValue(&TimerCounter, TIMER_COUNTER_0);
	xil_printf("NEON Multiplier Timer: %d \n\r",Value2);

    if (N<=8){
    	xil_printf("NEON Multiplier output\n\r");

    	//output
    	for (int i = 0; i < N; i++){
    		for (int j = 0; j < N; j++)
    			xil_printf("%d \t",D[i*N+j]);
    		xil_printf("\n\r");
    	}
    }
/* Verify correct output */

    if (!checkCorrect(C,D))
    	xil_printf("NEON multiplier matches classical multiplier\r\n");
    else
    	xil_printf("NEON multiplier does not match classical multiplier\r\n");
	return XST_SUCCESS;

}

void classicMatMult(u16 A[], u16 B[], u16 C[]){
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] = C[i*N+j] + A[i*N+k]*B[k*N+j];

}

/* insert neonMatMult() here */
/*======================================================*/

#if (N % 8) == 0

#define BLOCKSIZE 8
typedef uint16x8_t vec;
typedef uint16x8x4_t vecval;
typedef uint16x4x4_t vecval2;
#define vload(c, ptr) c = vld1q_u16(ptr);
#define vload_half(c, ptr) c = vld1_u16(ptr);
#define vstore(ptr, c) vst1q_u16(ptr, c);
#define vmla(c, a, b) c = vmlaq_u16(c, a, b);
#define vmlalane(c, a, b, n) c = vmlaq_lane_u16(c, a, b, n);
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


#if (N % 8) == 0
static
void neonMatMul_base(u16 A[], u16 B[], u16 C[])
{
    // Total registers:  24
    vecval vb1, vb2, vc1, vc2;


    vecval2 va0, va1, va2, va3;

    vload_half(va0.val[0], &A[0 * N]);
    vload_half(va0.val[1], &A[0 * N + 4]);
    vload_half(va0.val[2], &A[1 * N]);
    vload_half(va0.val[3], &A[1 * N + 4]);

    vload_half(va1.val[0], &A[2 * N]);
    vload_half(va1.val[1], &A[2 * N + 4]);
    vload_half(va1.val[2], &A[3 * N]);
    vload_half(va1.val[3], &A[3 * N + 4]);

    vload_half(va2.val[0], &A[4 * N]);
    vload_half(va2.val[1], &A[4 * N + 4]);
    vload_half(va2.val[2], &A[5 * N]);
    vload_half(va2.val[3], &A[5 * N + 4]);

    vload_half(va3.val[0], &A[6 * N]);
    vload_half(va3.val[1], &A[6 * N + 4]);
    vload_half(va3.val[2], &A[7 * N]);
    vload_half(va3.val[3], &A[7 * N + 4]);


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

    vmlalane(vc1.val[0], vb1.val[0], va0.val[0], 0);
    vmlalane(vc1.val[0], vb1.val[1], va0.val[0], 1);
    vmlalane(vc1.val[0], vb1.val[2], va0.val[0], 2);
    vmlalane(vc1.val[0], vb1.val[3], va0.val[0], 3);

    vmlalane(vc1.val[0], vb2.val[0], va0.val[1], 0);
    vmlalane(vc1.val[0], vb2.val[1], va0.val[1], 1);
    vmlalane(vc1.val[0], vb2.val[2], va0.val[1], 2);
    vmlalane(vc1.val[0], vb2.val[3], va0.val[1], 3);

    vmlalane(vc1.val[1], vb1.val[0], va0.val[2], 0);
    vmlalane(vc1.val[1], vb1.val[1], va0.val[2], 1);
    vmlalane(vc1.val[1], vb1.val[2], va0.val[2], 2);
    vmlalane(vc1.val[1], vb1.val[3], va0.val[2], 3);

    vmlalane(vc1.val[1], vb2.val[0], va0.val[3], 0);
    vmlalane(vc1.val[1], vb2.val[1], va0.val[3], 1);
    vmlalane(vc1.val[1], vb2.val[2], va0.val[3], 2);
    vmlalane(vc1.val[1], vb2.val[3], va0.val[3], 3);

    vmlalane(vc1.val[2], vb1.val[0], va1.val[0], 0);
    vmlalane(vc1.val[2], vb1.val[1], va1.val[0], 1);
    vmlalane(vc1.val[2], vb1.val[2], va1.val[0], 2);
    vmlalane(vc1.val[2], vb1.val[3], va1.val[0], 3);

    vmlalane(vc1.val[2], vb2.val[0], va1.val[1], 0);
    vmlalane(vc1.val[2], vb2.val[1], va1.val[1], 1);
    vmlalane(vc1.val[2], vb2.val[2], va1.val[1], 2);
    vmlalane(vc1.val[2], vb2.val[3], va1.val[1], 3);

    vmlalane(vc1.val[3], vb1.val[0], va1.val[2], 0);
    vmlalane(vc1.val[3], vb1.val[1], va1.val[2], 1);
    vmlalane(vc1.val[3], vb1.val[2], va1.val[2], 2);
    vmlalane(vc1.val[3], vb1.val[3], va1.val[2], 3);

    vmlalane(vc1.val[3], vb2.val[0], va1.val[3], 0);
    vmlalane(vc1.val[3], vb2.val[1], va1.val[3], 1);
    vmlalane(vc1.val[3], vb2.val[2], va1.val[3], 2);
    vmlalane(vc1.val[3], vb2.val[3], va1.val[3], 3);

    vmlalane(vc2.val[0], vb1.val[0], va2.val[0], 0);
    vmlalane(vc2.val[0], vb1.val[1], va2.val[0], 1);
    vmlalane(vc2.val[0], vb1.val[2], va2.val[0], 2);
    vmlalane(vc2.val[0], vb1.val[3], va2.val[0], 3);

    vmlalane(vc2.val[0], vb2.val[0], va2.val[1], 0);
    vmlalane(vc2.val[0], vb2.val[1], va2.val[1], 1);
    vmlalane(vc2.val[0], vb2.val[2], va2.val[1], 2);
    vmlalane(vc2.val[0], vb2.val[3], va2.val[1], 3);

    vmlalane(vc2.val[1], vb1.val[0], va2.val[2], 0);
    vmlalane(vc2.val[1], vb1.val[1], va2.val[2], 1);
    vmlalane(vc2.val[1], vb1.val[2], va2.val[2], 2);
    vmlalane(vc2.val[1], vb1.val[3], va2.val[2], 3);

    vmlalane(vc2.val[1], vb2.val[0], va2.val[3], 0);
    vmlalane(vc2.val[1], vb2.val[1], va2.val[3], 1);
    vmlalane(vc2.val[1], vb2.val[2], va2.val[3], 2);
    vmlalane(vc2.val[1], vb2.val[3], va2.val[3], 3);

    vmlalane(vc2.val[2], vb1.val[0], va3.val[0], 0);
    vmlalane(vc2.val[2], vb1.val[1], va3.val[0], 1);
    vmlalane(vc2.val[2], vb1.val[2], va3.val[0], 2);
    vmlalane(vc2.val[2], vb1.val[3], va3.val[0], 3);

    vmlalane(vc2.val[2], vb2.val[0], va3.val[1], 0);
    vmlalane(vc2.val[2], vb2.val[1], va3.val[1], 1);
    vmlalane(vc2.val[2], vb2.val[2], va3.val[1], 2);
    vmlalane(vc2.val[2], vb2.val[3], va3.val[1], 3);

    vmlalane(vc2.val[3], vb1.val[0], va3.val[2], 0);
    vmlalane(vc2.val[3], vb1.val[1], va3.val[2], 1);
    vmlalane(vc2.val[3], vb1.val[2], va3.val[2], 2);
    vmlalane(vc2.val[3], vb1.val[3], va3.val[2], 3);

    vmlalane(vc2.val[3], vb2.val[0], va3.val[3], 0);
    vmlalane(vc2.val[3], vb2.val[1], va3.val[3], 1);
    vmlalane(vc2.val[3], vb2.val[2], va3.val[3], 2);
    vmlalane(vc2.val[3], vb2.val[3], va3.val[3], 3);

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
static
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


void neonMatMult(u16 A[], u16 B[], u16 C[])
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

/*======================================================*/

int checkCorrect(u16 A[], u16 B[]){
	int result = 0;
	for (int i = 0; i < N*N; i++)
		if (A[i] != B[i])
			result = 1;
	return result;
}

