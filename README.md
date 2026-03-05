Hạng,Thành viên,Thuật toán (Model),Tham số,Kết quả RMSE,Đánh giá so với Baseline (18.88)
🥇,Hưng,Gradient Boosting,Mặc định,9.7825,"Tốt nhất, vượt trội Baseline"
🥈,Khoa,Random Forest,Mặc định,10.2716,"Rất tốt, vượt Baseline"
🥉,Thắng,Support Vector (SVR),Mặc định,(10.6714),"Khá tốt, vượt Baseline"
4,Khôi,Decision Tree,Mặc định,15.1239,"Tạm ổn, nhỉnh hơn Baseline một chút"
(Mốc),Điền (Leader),Baseline (Đoán trung bình),N/A,18.8873,Chỉ đoán bừa

ĐÁNH GIÁ KẾT QUẢ SPRINT 1:

Mô hình Gradient Boosting (do Hưng phụ trách) đang dẫn đầu với mức sai số thấp nhất (RMSE ~9.78). Điều này cho thấy thuật toán Boosting học các mẫu dữ liệu (pattern) của sinh viên rất hiệu quả.

Decision Tree (do Khôi phụ trách) kém hiệu quả nhất (15.12) do hiện tượng Overfitting (mô hình học vẹt dữ liệu Train nhưng dự đoán kém trên tập Test), vì chúng ta đang để tham số max_depth=None (cây phát triển vô hạn).

Chưa có mô hình nào đạt được mục tiêu RMSE < 8. Tuy nhiên, tất cả các mô hình đều chiến thắng thuyết phục mô hình Baseline (18.88). Điều này chứng minh rằng các đặc trưng như study_hours, class_attendance... thực sự có sức mạnh dự đoán điểm số, thay vì chỉ là dữ liệu nhiễu.

HƯỚNG PHÁT TRIỂN TIẾP THEO (SPRINT 2):
Nhóm sẽ ngừng sử dụng Decision Tree và dồn lực vào việc tinh chỉnh tham số (Hyperparameter Tuning) cho 2 mô hình tiềm năng nhất là Gradient Boosting và Random Forest để ép sai số RMSE xuống dưới 8 như cam kết.

