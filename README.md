# Matmul A * X (khối Hyperconnection)

Mô hình C: `X (bf16, đã có trong thanh ghi x_i) -> Matmul Y = A * X -> Store -> RAB (o_y)`.

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
2. Các bước chạy (Windows, từ lúc mở PowerShell)
Bước 1. Mở PowerShell
Nhấn phím Windows, gõ PowerShell, nhấn Enter.

Bước 2. Kiểm tra đã có gcc, make và git
gcc --version
mingw32-make --version
git --version
Mỗi lệnh phải in ra số phiên bản. Nếu lệnh nào báo is not recognized thì làm theo mục 3 (cài công cụ) rồi quay lại.

Bước 3. Lấy code từ GitHub
git clone https://github.com/htranngochuy/Deepseek-.git
Bước 4. Vào thư mục gốc của project
cd Deepseek-
Bước 5. Kiểm tra đang ở đúng thư mục
dir
Phải thấy Makefile, include, source, test.

Bước 6. Biên dịch và chạy
mingw32-make run

