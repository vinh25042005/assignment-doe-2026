import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')

# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
df = pd.read_csv('data/mlc_churn.csv')
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

#loại bỏ biến đa cộng tuyến 
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_auto = [column for column in upper.columns if any(upper[column] > 0.95)]
cols_to_drop = to_drop_auto + ['churn']

X = df.drop(columns=cols_to_drop)
y = df['churn']

# 2. THÍ NGHIỆM CRD (k=3, 5, 10; lặp 10 lần)
print("Bắt đầu chạy Thí nghiệm CRD")
k_values = [3, 5, 10]
crd_results = []
RANDOM_SEED = 1234

for k in k_values:
    print(f" -> Đang huấn luyện với k = {k}...")
    # Chia fold stratified và lặp 10 lần
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        crd_results.append({'k': k, 'f1_score': f1})

# Lưu kết quả
os.makedirs('results/crd', exist_ok=True)
df_crd = pd.DataFrame(crd_results)
df_crd.to_csv('results/crd/crd_results.csv', index=False)
print("Đã lưu file dữ liệu kết quả tại: results/crd/crd_results.csv\n")

# 3. PHÂN TÍCH THỐNG KÊ
print("="*50)
print("KẾT QUẢ PHÂN TÍCH THỐNG KÊ")
print("="*50)

# So sánh phương sai bằng Levene Test
f1_k3 = df_crd[df_crd['k'] == 3]['f1_score']
f1_k5 = df_crd[df_crd['k'] == 5]['f1_score']
f1_k10 = df_crd[df_crd['k'] == 10]['f1_score']

# Chạy các kiểm định
stat, p_levene = stats.levene(f1_k3, f1_k5, f1_k10)
model = ols('f1_score ~ C(k)', data=df_crd).fit()
tukey = pairwise_tukeyhsd(endog=df_crd['f1_score'], groups=df_crd['k'], alpha=0.05)

#lưu kết quả phân tích thống kê
with open('results/crd/statistical_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("==================================================\n")
    f.write("KẾT QUẢ PHÂN TÍCH THỐNG KÊ - THÍ NGHIỆM CRD\n")
    f.write("==================================================\n\n")
    
    f.write("[1] KIỂM ĐỊNH LEVENE (So sánh phương sai)\n")
    f.write(f"p-value = {p_levene:.4f}\n")
    if p_levene > 0.05:
        f.write("-> p-value > 0.05: Phương sai giữa các nhóm k đồng nhất.\n\n")
    else:
        f.write("-> p-value < 0.05: Phương sai giữa các nhóm k khác biệt.\n\n")
        
    f.write("[2] BẢNG PHÂN TÍCH OLS (Giá trị trung bình và Khoảng tin cậy)\n")
    f.write(str(model.summary().tables[1]) + "\n\n")
    
    f.write("[3] BẢNG PHÂN TÍCH TUKEY HSD (So sánh cặp)\n")
    f.write(str(tukey.summary()) + "\n")

print("Đã lưu các bảng thống kê tại: results/crd/statistical_analysis.txt")

# Vẽ và lưu đồ thị TukeyHSD
tukey.plot_simultaneous()
plt.title("Tukey HSD: So sánh F1-Score theo số fold (k)")
plt.tight_layout()
plt.savefig('results/crd/crd_tukey_plot.png', dpi=300)
print("\nĐã lưu biểu đồ Tukey tại: results/crd/crd_tukey_plot.png")