#include "store.h"

void store_lc_o_y(registers_format *regs, RAB_format *RAB, int token_id) {
    memcpy(RAB->o_y[token_id], regs->y_l, sizeof(regs->y_l));
}
