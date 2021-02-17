# Matrix Multiplication Kernel in ARMv7 and ARMv8

## Kernel 

Kernel size: 

- Multiple of 4
- Multiple of 8

I can't implememnt multiple of 16, simply run out of 32 registers for 16-bit types.

## Performance Notes

- ARMv7 I have not optimized vector load. 

- ARMv8 is optimized. 

## Result 

Here I only show result of ARMv8. Who cares to use ARMv7 nowadays. 

Compile flags:
```bash
gcc -o matmul matmul.c libpapi.a -g3 -O3 -fno-tree-vectorize -Wall -Wextra -Wpedantic
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

| Size     	| 64         	| 72           	| 128          	| 132          	|           256 	|           260 	|
|----------	|------------	|--------------	|--------------	|--------------	|--------------:	|--------------:	|
| Pure C   	| 872,426.02 	| 1,232,119.77 	| 6,639,902.38 	| 7,362,135.01 	| 51,960,426.14 	| 55,365,862.46 	|
| Kernel 4 	|      na      	|   112,541.69 	|     na       	| 1,002,903.70 	|      na         	|  8,194,332.11 	|
| Kernel 8 	|  81,586.58 	|      na      	|   666,325.63 	|     na       	|  5,413,270.20 	|      na         	|
| Ratio    	| 10.69      	| 10.95        	| 9.96         	| 7.34         	| 9.60          	| 6.76          	|


