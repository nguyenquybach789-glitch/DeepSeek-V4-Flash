#include "kernel.h"

void matmul_A_X(registers_format *reg) {
    for (int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for (int i = 0; i < N_HC; i++)
            sum += reg->A_l[i] * bf16_to_fp32(reg->x_i[i][j]);
        reg->y_l[j] = fp32_to_bf16(sum);
    }
}
