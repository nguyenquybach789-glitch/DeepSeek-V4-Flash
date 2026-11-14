# Matmul A × X và Store (khối Hyperconnection)

Module này tính `Y = A × X` rồi ghi kết quả vào bộ nhớ đệm RAB.

- `A` là vector fp32 kích thước `[N_HC]`.
- `X` là ma trận bf16 kích thước `[N_HC][D_MODEL]`, đã nằm sẵn trong thanh ghi `REG.x_i`.
- `Y` là vector bf16 kích thước `[D_MODEL]`, lưu tại `REG.y_l` rồi được copy sang `RAB.o_y`.

Module này **không có LOAD**. Trong `test/main.c`, dữ liệu X được tạo ngẫu nhiên rồi copy thẳng vào `REG.x_i` để giả lập bước LOAD.

Luồng dữ liệu:

```
X (test tạo ngẫu nhiên, bf16)
   │  memcpy (giả lập LOAD)
   ▼
REG.x_i  ──matmul_A_X(&REG)──▶  REG.y_l
                                   │  store_lc_o_y(&REG, &RAB, 0)
                                   ▼
                              RAB.o_y[0]
```

---

## 1. Cấu trúc thư mục

```
<thư-mục-gốc>/
├── Makefile                 luật biên dịch và chạy
├── README.md                tài liệu này
├── .gitignore               bỏ qua file *.exe
├── include/                 khai báo (header)
│   ├── config.h             hằng số: N_HC, D_MODEL, N_TOKEN, RAB_TOKENS
│   ├── data_type.h          kiểu bf16 và hai hàm chuyển đổi
│   ├── data_structure.h     RAB_format, registers_format
│   ├── kernel.h             khai báo matmul_A_X
│   └── store.h              khai báo store_lc_o_y
├── source/                  cài đặt (mọi file .c ở đây được biên dịch)
│   ├── data_type.c          bf16_to_fp32, fp32_to_bf16
│   ├── kernel.c             matmul_A_X
│   └── store.c              store_lc_o_y
└── test/
    └── main.c               chương trình chạy thử
```

Quy tắc chung: file trong `include/` chỉ khai báo, file trong `source/` chứa phần thân hàm. Mỗi file trong `source/` đều có file `.h` tương ứng trong `include/` (`data_type.c` ↔ `data_type.h`, `kernel.c` ↔ `kernel.h`, `store.c` ↔ `store.h`).

---

## 2. Chuẩn bị môi trường

Cần có `gcc` (trình biên dịch C) và `make`.

### Windows

1. Cài MinGW-w64 (ví dụ qua MSYS2 hoặc trực tiếp từ winlibs.com) để có `gcc` và `mingw32-make`.
2. Thêm thư mục `bin` của MinGW vào biến môi trường `PATH`.
3. Mở PowerShell mới và kiểm tra:

```powershell
gcc --version
mingw32-make --version
```

Mỗi lệnh phải in ra số phiên bản. Nếu báo `is not recognized`, hãy kiểm tra lại `PATH` rồi mở lại PowerShell.

### Linux / macOS

```sh
gcc --version
make --version
```

Trên macOS, nếu chưa có `gcc`, chạy `xcode-select --install`.

---

## 3. Đưa code vào máy

1. Giải nén file zip vào một thư mục, ví dụ `C:\Users\<tên>\Desktop\matmul_project`.
2. Mở PowerShell (Windows) hoặc Terminal (Linux/macOS).
3. Di chuyển vào thư mục gốc, là thư mục có `Makefile`:

```powershell
cd C:\Users\<tên>\Desktop\matmul_project
dir
```

Phải thấy `Makefile`, `include`, `source`, `test`. Nếu không thấy, bạn đang đứng sai thư mục.

---

## 4. Biên dịch và chạy

Trên Windows:

```powershell
mingw32-make run
```

Trên Linux/macOS:

```sh
make run
```

Kết quả mong đợi:

```
15.562500 15.375000 15.187500 7.500000
```

Đây là 4 giá trị đầu của `Y`. Với cùng seed `srand(1)` và cùng `A`, bạn phải thấy đúng dãy số này. Nếu khác, kiểm tra lại các bước sửa code ở mục 6.

Các lệnh khác:

| Lệnh | Tác dụng |
|---|---|
| `make` hoặc `mingw32-make` | Chỉ biên dịch, tạo `test/app.exe` |
| `make run` | Biên dịch (nếu cần) rồi chạy |
| `make clean` | Xóa file `test/app.exe` |

`Makefile` tự biên dịch lại khi bạn sửa bất kỳ file `.c` hoặc `.h` nào.

---

## 5. Đọc hiểu code

### 5.1. `include/config.h`

Chứa các hằng số dùng chung:

- `N_HC = 4`: số luồng (hàng) của X và số phần tử của A.
- `D_MODEL = 4096`: số cột của X, cũng là độ dài của Y.
- `N_TOKEN`, `RAB_TOKENS`: số token. Hiện đặt bằng 1.

Muốn đổi kích thước, chỉ sửa ở đây.

### 5.2. `include/data_type.h` và `source/data_type.c`

bf16 được lưu dưới dạng `uint16_t`, gói trong struct `bf16`:

- `bf16_to_fp32(bf16 b)`: đưa 16 bit lên 16 bit cao của float32, phần thấp bằng 0. Đây là phép đổi chính xác, không mất dữ liệu.
- `fp32_to_bf16(float f)`: lấy 16 bit cao của float32. Đây là **cắt bit (truncation)**, không làm tròn, nên có thể sai lệch nhỏ so với PyTorch.

### 5.3. `include/data_structure.h`

```c
typedef struct {
    bf16 o_y[RAB_TOKENS][D_MODEL];    /* RAB: lưu kết quả Y */
} RAB_format;

typedef struct {
    float A_l[N_HC];                  /* hệ số A, fp32 */
    bf16  x_i[N_HC][D_MODEL];         /* đầu vào X, bf16 */
    bf16  y_l[D_MODEL];               /* kết quả Y, bf16 */
} registers_format;
```

`registers_format` mô phỏng các thanh ghi cục bộ. `RAB_format` mô phỏng bộ nhớ đệm.

### 5.4. `source/kernel.c`: phép nhân `A × X`

```c
void matmul_A_X(registers_format *reg) {
    for (int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for (int i = 0; i < N_HC; i++)
            sum += reg->A_l[i] * bf16_to_fp32(reg->x_i[i][j]);
        reg->y_l[j] = fp32_to_bf16(sum);
    }
}
```

Với mỗi cột `j`:

`Y[j] = A[0]·X[0][j] + A[1]·X[1][j] + ... + A[N_HC-1]·X[N_HC-1][j]`

Tích lũy bằng fp32, sau đó đổi kết quả sang bf16 và lưu vào `y_l[j]`.

### 5.5. `source/store.c`: ghi vào RAB

```c
void store_lc_o_y(registers_format *regs, RAB_format *RAB, int token_id) {
    memcpy(RAB->o_y[token_id], regs->y_l, sizeof(regs->y_l));
}
```

Hàm này copy toàn bộ `y_l` (4096 phần tử bf16, tức 8 KB) vào hàng `token_id` của RAB.

### 5.6. `test/main.c`: trình tự chạy

1. Tạo X ngẫu nhiên với `srand(1)`, lưu vào `DDR4_x` (bf16).
2. Khai báo `A = {0.55, 0.48, 0.62, 0.57}` và copy vào `REG.A_l`.
3. Copy `DDR4_x` vào `REG.x_i` (giả lập LOAD).
4. Gọi `matmul_A_X(&REG)` để tính `REG.y_l`.
5. Gọi `store_lc_o_y(&REG, &RAB, 0)` để ghi vào `RAB.o_y[0]`.
6. In 4 giá trị đầu của `RAB.o_y[0]`.

---

## 6. Thực hành: tự sửa và kiểm tra

### Bài 1: kiểm tra bằng số đã biết

Nếu mọi `X[i][j] = 1`, thì `Y[j] = A[0] + A[1] + A[2] + A[3] = 0.55 + 0.48 + 0.62 + 0.57 = 2.22`. Kết quả sẽ hơi khác 2.22 vì bf16 chỉ có 8 bit mantissa.

Mở `test/main.c`, thay vòng lặp tạo X bằng:

```c
for (int i = 0; i < N_HC; i++)
    for (int j = 0; j < D_MODEL; j++)
        DDR4_x[i][j] = fp32_to_bf16(1.0f);
```

Chạy lại `mingw32-make run`. Bạn phải thấy các giá trị gần `2.21875`. Sau đó hoàn nguyên lại code gốc.

### Bài 2: thử một ma trận nhỏ

Đặt `A = {1, 0, 0, 0}`. Khi đó `Y[j] = X[0][j]`, nên `RAB.o_y[0]` phải bằng đúng hàng 0 của X.

### Bài 3: đổi kích thước

Sửa `D_MODEL` trong `include/config.h` thành `8`, rồi chạy lại. Chương trình vẫn phải chạy và in 4 số đầu.

### Bài 4: đổi cách làm tròn

Trong `source/data_type.c`, thay `fp32_to_bf16` bằng phiên bản làm tròn về số gần nhất (round-to-nearest-even). Sau đó so sánh kết quả với bản cắt bit để thấy sai khác.

---

## 7. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách sửa |
|---|---|
| `gcc is not recognized` | Chưa cài hoặc chưa thêm MinGW vào `PATH`. Xem mục 2. |
| `mingw32-make is not recognized` | Thư mục `bin` của MinGW chưa có trong `PATH`, hoặc bạn dùng `make` thay vì `mingw32-make` trên Windows. |
| `No rule to make target` hoặc `No such file` | Bạn đang đứng sai thư mục. Chạy `dir` để kiểm tra `Makefile`. |
| `undefined reference to ...` | Thiếu file `.c` trong `source/`, hoặc tên hàm trong `.c` không khớp với `.h`. |
| `conflicting types` | Chữ ký hàm trong `.c` khác với `.h`. |
| Kết quả khác `15.562500 ...` | Đã sửa `A`, `srand`, hoặc vòng lặp tạo X trong `test/main.c`. Hoàn nguyên lại và chạy lại. |

---

## 8. Ghi chú

- Module này chỉ gồm Matmul và Store. LOAD, Flatten và RMSNorm nằm ở module khác.
- Phần ghi vào RAB dùng `memcpy` trực tiếp. Khi tích hợp vào nền tảng phần cứng, bạn có thể thay bằng hàm bus riêng.
- Accumulate bằng fp32, kết quả được làm tròn xuống bf16 bằng cách cắt bit, nên giá trị có thể lệch tối đa khoảng 0,8% (tương ứng 1/128) so với tính toán toán học chính xác.
