import pandas as pd
from ydata_profiling import ProfileReport

# 1. Load bộ dữ liệu
# Đảm bảo file data.csv nằm đúng thư mục
file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df = pd.read_csv(file_path)

# 2. Khởi tạo báo cáo với cấu hình ĐÚNG VỊ TRÍ
profile = ProfileReport(
    df,
    title="EWS Data Analysis Report - Verified Version",
    explorative=True,
    # Chuyển dark_mode vào trong html
    html={
        "dark_mode": True,
        "full_width": True
    },
    # Chuyển author vào trong dataset
    dataset={
        "author": "Le Thanh Dien - Group 3",
        "description": "Early Warning System Analysis",
        "copyright_holder": "FPT University"
    }
)

# 3. Xuất ra file HTML
output_file = "EWS_Profile_Report_v2.html"
profile.to_file(output_file)

print(f"SUCCESS: Report updated and saved to {output_file}!")