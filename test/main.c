#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_structure.h"
#include "data_type.h"
#include "kernel.h"
#include "store.h"
#include "config.h"

static bf16 DDR4_x[N_HC][D_MODEL];
static RAB_format RAB;
static registers_format REG;

int main(void)
{
    srand(1);
    for (int i = 0; i < N_HC; i++)
        for (int j = 0; j < D_MODEL; j++)
            DDR4_x[i][j] = fp32_to_bf16((float)(rand() % 100) / 10.0f);

    /* Hệ số A (fp32) */
    const float A[N_HC] = { 0.55f, 0.48f, 0.62f, 0.57f };
    memcpy(REG.A_l, A, sizeof(A));

    /* Dữ liệu test được đưa thẳng vào thanh ghi x_i (không có module load) */
    memcpy(REG.x_i, DDR4_x, sizeof(REG.x_i));

    matmul_A_X(&REG);
    store_lc_o_y(&REG, &RAB, 0);

    for (int j = 0; j < 4; j++)
        printf("%.6f ", bf16_to_fp32(RAB.o_y[0][j]));
    printf("\n");
    return 0;
}
