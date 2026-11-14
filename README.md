# Matmul A * X (khối Hyperconnection)

Mô hình C: `X (bf16, đã có trong thanh ghi x_i) -> Matmul Y = A * X -> Store -> RAB (o_y)`.
Module này không có LOAD: dữ liệu X được test đưa thẳng vào `REG.x_i`.

## Cấu trúc thư mục

```
<thư-mục-gốc>/
├── Makefile
├── include/
│   ├── config.h           hằng số
│   ├── data_type.h        kiểu bf16 và chuyển đổi bf16 <-> fp32
│   ├── data_structure.h   RAB_format, registers_format
│   ├── kernel.h           matmul_A_X
│   └── store.h            store_lc_o_y (thanh ghi -> RAB)
├── source/
│   ├── data_type.c
│   ├── kernel.c
│   └── store.c
└── test/
    └── main.c
```

## Công thức

Với mỗi cột `j`: `Y[j] = sum(i = 0..N_HC-1) A[i] * bf16_to_fp32(X[i][j])`.
Tích lũy theo fp32, kết quả đổi về bf16 và lưu vào `regs.y_l`.

## Chạy

```sh
make run
```

Trên Windows dùng `mingw32-make run`.
