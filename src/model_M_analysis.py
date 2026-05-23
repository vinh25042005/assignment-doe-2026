import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. ĐỌC VÀ MÃ HÓA DỮ LIỆU
df = pd.read_csv('data/mlc_churn.csv')

# Mã hóa biến mục tiêu churn (yes=1, no=0) để tính F1
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

# Mã hóa các biến dạng chuỗi (categorical) sang số
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 2. PHÂN TÍCH TƯƠNG QUAN & CHỌN BIẾN
# ==========================================
# 2. PHÂN TÍCH TƯƠNG QUAN & CHỌN BIẾN
# ==========================================
print("Đang phân tích tương quan các biến...")
corr_matrix = df.corr().abs()

# Lấy nửa trên của ma trận tương quan
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Tìm các cặp biến có độ tương quan > 0.95
to_drop_auto = []
high_corr_details = []

for column in upper.columns:
    highly_correlated_with = upper.index[upper[column] > 0.95].tolist()
    for row in highly_correlated_with:
        corr_value = upper.loc[row, column]
        to_drop_auto.append(column)
        high_corr_details.append(f"   - Bỏ '{column}' (tương quan = {corr_value:.4f} với '{row}')")

# Loại bỏ các biến trùng lặp trong list
to_drop_auto = list(set(to_drop_auto))
print(f"Các biến có tương quan > 0.95 phát hiện được: {to_drop_auto}")

# Quyết định loại bỏ: Dựa vào ý nghĩa thực tiễn
cols_to_drop = to_drop_auto + ['churn']
X = df.drop(columns=cols_to_drop)
y = df['churn']
print("Đã loại bỏ các biến nhiễu/đa cộng tuyến.")

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

# 4. LƯU KẾT QUẢ VÀO FOLDER RIÊNG
print("Đang lưu kết quả...")
os.makedirs('results/baseline', exist_ok=True)

# Lưu hiệu năng F1-Score
df_baseline = pd.DataFrame([{'model': 'Baseline Random Forest (M)', 'f1_score': f1_M}])
df_baseline.to_csv('results/baseline/baseline_results.csv', index=False)

# Lưu độ quan trọng của các biến (Feature Importance)
importances = model_M.feature_importances_
df_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
df_importance.to_csv('results/baseline/feature_importances.csv', index=False)

print("Đã lưu kết quả Mô hình M tại: results/baseline/")

# Lưu danh sách các biến đã loại bỏ ra file text
with open('results/baseline/dropped_features.txt', 'w', encoding='utf-8') as f:
    f.write("DANH SÁCH CÁC BIẾN ĐÃ LOẠI BỎ KHỎI TẬP HUẤN LUYỆN:\n")
    f.write("==================================================\n\n")
    
    f.write("1. Các biến bị loại do có độ tương quan quá cao (> 0.95) gây đa cộng tuyến:\n")
    if len(high_corr_details) > 0:
        for detail in high_corr_details:
            f.write(detail + "\n")
    else:
        f.write("   - Không có biến nào.\n")
        
    f.write("\n2. Biến mục tiêu (Target variable) được tách riêng để dự đoán:\n")
    f.write("   - churn\n")
        
print("Đã lưu chi tiết danh sách biến bị loại tại: results/baseline/dropped_features.txt")