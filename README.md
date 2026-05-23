# Phân Tích Thực Nghiệm Mô Hình Machine Learning: mlc_churn

Dự án thực hiện quy trình thiết kế và phân tích thực nghiệm nhằm đánh giá hiệu năng của mô hình Random Forest trên bộ dữ liệu `mlc_churn`. Mục tiêu là xác định ảnh hưởng của số lượng fold (k) và độ sâu cây (max_depth) đến độ đo F1-Score.

## Cấu trúc thư mục
```text

.
├── data/                   # Bộ dữ liệu gốc (mlc_churn.csv)
├── src/
│   ├── model_M_analysis.py # Thí nghiệm xây dựng mô hình cơ sở
│   ├── crd_experiment.py   # Thí nghiệm 1: Thiết kế CRD
│   └── crfd_experiment.py  # Thí nghiệm 2: Thiết kế CRFD
├── results/
│   ├── baseline/           # Kết quả mô hình M (CSV, feature importance, log)
│   ├── crd/                # Kết quả thí nghiệm 1 (CSV, biểu đồ Tukey, log)
│   └── crfd/               # Kết quả thí nghiệm 2 (CSV, biểu đồ tương tác, log)
├── README.md               # Tài liệu hướng dẫn
```
## Nội dung thí nghiệm
Thí nghiệm 1 (CRD): Đánh giá ảnh hưởng của số lượng fold (k = 3, 5, 10) đối với F1-Score bằng kiểm định Levene, OLS và Tukey HSD.

Thí nghiệm 2 (CRFD): Phân tích tương tác giữa số fold (k) và độ sâu cây (max_depth) thông qua ANOVA 2 chiều và mô hình tuyến tính OLS.

## Hướng dẫn thí nghiệm

Để tái hiện kết quả, thực hiện các bước sau:

Cài đặt thư viện: 
```bash
pip install -r requirements.txt
```

Thực thi phân tích: Chạy file chính trong thư mục src/:

```bash 
python src/model_analysis.py
```
Kết quả sẽ tự động lưu vào các thư mục tương ứng trong results/