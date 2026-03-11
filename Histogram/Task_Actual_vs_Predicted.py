import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. NẠP DỮ LIỆU ĐÃ CHUẨN HÓA (File ông đã tạo ở bước trước)
# ==========================================
file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df = pd.read_csv(file_path)

print("="*70)
print("ĐANG TẠO BIỂU ĐỒ ACTUAL VS PREDICTED")
print("="*70)

# ==========================================
# 2. CHIA DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH
# ==========================================
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# Chia 90% Train, 10% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Lấy kết quả dự đoán
y_pred = model.predict(X_test)

# ==========================================
# 3. VẼ BIỂU ĐỒ TRỰC QUAN
# ==========================================
print("\nĐang vẽ đồ thị phân tán (Scatter Plot)...")
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# Vẽ các điểm dự đoán (Màu xanh)
plt.scatter(y_test, y_pred, alpha=0.5, color='#3498db', edgecolors='k', s=50, label='Sinh viên (Predicted vs Actual)')

# Vẽ đường thẳng lý tưởng y = x (Màu đỏ đứt nét)
# Bất kỳ sinh viên nào nằm trên đường này tức là máy dự đoán đúng 100%
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Đường dự đoán hoàn hảo (y=x)')

# Tính toán các chỉ số đánh giá để in thẳng lên hình
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Gắn nhãn chỉ số góc trên cùng bên trái
bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#7f8c8d", lw=1.5, alpha=0.9)
plt.text(0.05, 0.95, f'$R^2$ Score: {r2:.4f}\nRMSE: {rmse:.2f}',
         transform=plt.gca().transAxes, fontsize=13, fontweight='bold',
         verticalalignment='top', bbox=bbox_props)

# Tinh chỉnh giao diện
plt.title('Model Performance: Actual vs. Predicted Analysis', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Điểm thi Thực tế (Ground Truth)', fontsize=13, fontweight='bold')
plt.ylabel('Điểm thi Dự đoán (Model Output)', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=12, frameon=True)

plt.tight_layout()
image_name = 'Actual_vs_Predicted.png'
plt.savefig(image_name, dpi=300)
print(f"[THÀNH CÔNG] Đã lưu biểu đồ thành file '{image_name}'.")