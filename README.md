# Matrix Multiplication Kernel in ARMv7 and ARMv8

## Kernel 

Kernel size: 

- 4x4
- 8x8
- 16x16
- 32x32
- 32x8

Conclusion: Speed of matrix multiplication pretty much depend on matrix size, thus one can customize the kernel to best fit the matrix multilication.



## Performance Notes

- ARMv7 I have not optimized vector load. 

- ARMv8 is optimized. 

## Result 

Here I only show result of ARMv8. Who cares to use ARMv7 nowadays. 

Compile flags:
```bash
gcc -o matmul matmul.c libpapi.a -g3 -O3 -fno-tree-vectorize -Wall -Wextra -Wpedantic
clang -o matmul matmul.c libpapi.a -g3 -O3 -fno-tree-vectorize -Wall -Wextra -Wpedantic
```

**Notes:** I put nasty `-fno-tree-vectorize` to show the contrast between vectorize and non-vectorize code.

```bash
$  lscpu

Architecture:                    aarch64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
CPU(s):                          4
On-line CPU(s) list:             0-3
Thread(s) per core:              1
Core(s) per socket:              4
Socket(s):                       1
Vendor ID:                       ARM
Model:                           3
Model name:                      Cortex-A72
Stepping:                        r0p3
CPU max MHz:                     1900.0000
CPU min MHz:                     600.0000
BogoMIPS:                        108.00
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Spec store bypass: Vulnerable
Vulnerability Spectre v1:        Mitigation; __user pointer sanitization
Vulnerability Spectre v2:        Vulnerable
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected
Flags:                           fp asimd evtstrm crc32 cpuid
``` 

| Size      	| 64         	| 72           	| 128          	| 132          	|           256 	|           260 	|
|-----------	|------------	|--------------	|--------------	|--------------	|--------------:	|--------------:	|
| Pure C    	| 872,426.02 	| 1,232,119.77 	| 6,639,902.38 	| 7,362,135.01 	| 51,960,426.14 	| 55,365,862.46 	|
| Kernel 4  	|            	|   112,541.69 	|              	| 1,002,903.70 	|               	|  8,194,332.11 	|
| Kernel 8  	|  81,586.58 	|              	|   666,325.63 	|              	|  5,413,270.20 	|               	|
| Ratio     	| 10.69      	| 10.95        	| 9.96         	| 7.34         	| 9.60          	| 6.76          	|
| Kernel 16 	|   82,980.6 	|              	|    704,353.4 	|              	|   5,668,996.2 	|               	|
| Ratio     	|       10.5 	|              	|          9.4 	|              	|           9.2 	|               	|
| Kernel 32 	|   79,647.7 	|              	|    743,547.8 	|              	|   5,900,377.0 	|               	|
| Ratio     	|       11.0 	|              	|          8.9 	|              	|           8.8 	|               	|


Note: It's funny that whenever I put compiler flags `-mtune=native` to `clang 11`, the result is worst. 
Similarly for GCC. 

It's a bit strange that kernel 32x32 slower than kernel 8x8 and kernel 16x16. I thought that kernel 32x32 consume more available registers and continuous load `vld1q_u16_x4` would save cycles. Hmm?

