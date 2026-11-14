#ifndef KERNEL_H
#define KERNEL_H

#include "data_structure.h"
#include "data_type.h"

/* Y = A * X: A là fp32 [N_HC], X và Y là bf16. Tích lũy fp32. */
void matmul_A_X(registers_format *reg);

#endif
