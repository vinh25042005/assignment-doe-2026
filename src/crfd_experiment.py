import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import warnings

warnings.filterwarnings('ignore')

# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
df = pd.read_csv('data/mlc_churn.csv')
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# loại bỏ biến đa cộng tuyến 
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_auto = [column for column in upper.columns if any(upper[column] > 0.95)]
cols_to_drop = to_drop_auto + ['churn']

X = df.drop(columns=cols_to_drop)
y = df['churn']

# 2. THÍ NGHIỆM CRFD (k & max_depth)
print("\nBắt đầu Thí nghiệm CRFD...")
k_values = [3, 5, 10]
max_depths = [3, 5, None]
crfd_results = []
RANDOM_SEED = 1234

os.makedirs('results/crfd', exist_ok=True)

for k in k_values:
    # Chia fold stratified, lặp 10 lần
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=RANDOM_SEED)
    
    for depth in max_depths:
        depth_str = "None" if depth is None else str(depth)
        print(f" -> Đang huấn luyện với k = {k}, max_depth = {depth_str}...")
        
        # Thiết lập mô hình với độ sâu tương ứng
        rf = RandomForestClassifier(max_depth=depth, random_state=RANDOM_SEED)
        
        for train_idx, test_idx in rskf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            
            crfd_results.append({
                'k': k, 
                'max_depth': depth_str, 
                'f1_score': f1
            })

# Lưu kết quả thô ra file CSV
df_crfd = pd.DataFrame(crfd_results)
df_crfd.to_csv('results/crfd/crfd_results.csv', index=False)
print("\nĐã lưu file dữ liệu kết quả tại: results/crfd/crfd_results.csv")

# 3. PHÂN TÍCH ANOVA 2 CHIỀU VÀ LƯU RA FILE TEXT

# Xây dựng mô hình OLS để đánh giá tác động của k, max_depth và sự tương tác giữa chúng
model_crfd = ols('f1_score ~ C(k) + C(max_depth) + C(k):C(max_depth)', data=df_crfd).fit()
anova_table = sm.stats.anova_lm(model_crfd, typ=2)

# LƯU BẢNG KẾT QUẢ RA FILE TEXT
with open('results/crfd/statistical_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("==================================================\n")
    f.write("KẾT QUẢ PHÂN TÍCH THỐNG KÊ - THÍ NGHIỆM CRFD\n")
    f.write("==================================================\n\n")
    
    f.write("[1] BẢNG PHÂN TÍCH PHƯƠNG SAI (ANOVA 2 CHIỀU)\n")
    f.write("Đánh giá ý nghĩa thống kê tương tác của k và max_depth:\n")
    f.write(str(anova_table) + "\n\n")
    
    f.write("[2] BẢNG HỆ SỐ OLS (Giá trị trung bình và Khoảng tin cậy)\n")
    f.write(str(model_crfd.summary().tables[1]) + "\n")

print("Đã lưu các bảng thống kê tại: results/crfd/statistical_analysis.txt")

# VẼ VÀ LƯU ĐỒ THỊ TƯƠNG TÁC 
fig, ax = plt.subplots(figsize=(8, 6))
# trace là đường biểu diễn các mức độ sâu (max_depth)
interaction_plot(x=df_crfd['k'], trace=df_crfd['max_depth'], response=df_crfd['f1_score'], 
                 colors=['red', 'blue', 'green'], markers=['D', '^', 'o'], ax=ax)

plt.title('Đồ thị Tương tác giữa Số fold (k) và Độ sâu cây (max_depth) lên F1-Score', fontsize=12)
plt.xlabel('Số fold (k)', fontsize=11)
plt.ylabel('Trung bình F1-Score', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('results/crfd/interaction_plot.png', dpi=300)
print("Đã lưu đồ thị tương tác tại: results/crfd/interaction_plot.png\n")