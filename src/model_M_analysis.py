import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. ĐỌC VÀ MÃ HÓA DỮ LIỆU
print("Đang đọc dữ liệu...")
df = pd.read_csv('data/mlc_churn.csv')

# Mã hóa biến mục tiêu churn (yes=1, no=0) để tính F1
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

# Mã hóa các biến dạng chuỗi (categorical) sang số
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 2. PHÂN TÍCH TƯƠNG QUAN & CHỌN BIẾN
print("Đang phân tích tương quan các biến...")
# Ma trận tương quan
corr_matrix = df.corr().abs()

# Tìm các cặp biến có độ tương quan  cao (> 0.95)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_auto = [column for column in upper.columns if any(upper[column] > 0.95)]
print(f"Các biến có tương quan > 0.95 được tự động phát hiện: {to_drop_auto}")

# Quyết định loại bỏ: Dựa vào ý nghĩa thực tiễn, các biến cước phí (charge) 
# được tính trực tiếp từ số phút gọi (minutes). Giữ cả 2 không mang lại thông tin mới.
cols_to_drop = to_drop_auto + ['churn']
X = df.drop(columns=cols_to_drop)
y = df['churn']
print(f"Đã loại bỏ các biến: {cols_to_drop[1:]}")

# 3. XÂY DỰNG MÔ HÌNH M (RANDOM FOREST)
print("Đang huấn luyện Mô hình M...")
# Chia tập train/test (stratify=y để giữ tỷ lệ mất cân bằng giữa rời mạng và không rời mạng)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1234)

#seed = 1234
model_M = RandomForestClassifier(random_state=1234)
model_M.fit(X_train, y_train)

# Dự đoán và tính F1-Score
y_pred = model_M.predict(X_test)
f1_M = f1_score(y_test, y_pred, pos_label=1)

print("-" * 40)
print(f"Hiệu năng của Mô hình M (F1-Score): {f1_M:.4f}")
