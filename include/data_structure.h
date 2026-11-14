#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H

#include <stdint.h>
#include "data_type.h"
#include "config.h"

/* Bộ nhớ đệm RAB: lưu kết quả Y */
typedef struct {
    bf16 o_y[RAB_TOKENS][D_MODEL];
} RAB_format;

/* Thanh ghi cục bộ: A, X và Y = A * X */
typedef struct {
    float A_l[N_HC];
    bf16  x_i[N_HC][D_MODEL];
    bf16  y_l[D_MODEL];
} registers_format;

#endif
