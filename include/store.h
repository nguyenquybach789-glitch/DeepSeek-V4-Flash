#ifndef STORE_H
#define STORE_H

#include "data_structure.h"

/* Thanh ghi -> RAB: ghi y_l vào RAB->o_y[token_id] */
void store_lc_o_y(
    registers_format *regs,
    RAB_format *RAB,
    int token_id);

#endif
