# Repository Guidelines

## Mục tiêu và cách cộng tác

Đồ án phân loại quả tắc theo hai nhãn độc lập: độ chín (`xanh/chín`) và độ tươi/héo (`tươi/héo`). Hai thành viên phụ trách HSV + SVM và CNN MobileNetV2; mỗi phương pháp dự đoán cả hai thuộc tính, sau đó fusion xác suất.

Hướng dẫn lý thuyết trước, code sau. Giải thích đầu vào, đầu ra và lý do lựa chọn. Chia thành bài nhỏ, đưa ví dụ tối thiểu rồi để người học thực hành. Giải thích nguyên nhân khi sửa lỗi. Không hoàn thành đồ án thay người học; chỉ triển khai phần được yêu cầu.

## Project Structure & Module Organization

Repository hiện chưa có mã nguồn. Cấu trúc dự kiến khi cần triển khai:

- `src/`: tiền xử lý, đặc trưng HSV, SVM, CNN và fusion.
- `app.py`: giao diện Gradio phục vụ demo báo cáo, tạo khi tới bước giao diện.
- `notebooks/`: thực hành và khảo sát dữ liệu.
- `tests/`: kiểm thử logic xử lý dữ liệu và đánh giá.
- `data/`: ảnh, mask và bảng nhãn; `outputs/`: mô hình, biểu đồ, báo cáo.

Không tạo sẵn các thư mục hoặc module khi chưa học tới bước tương ứng.

## Build, Test, and Development Commands

Chưa có dependency manifest hoặc entrypoint. Khi bắt đầu code:

- `python -m venv .venv`: tạo môi trường.
- `python -m pip install -r requirements.txt`: cài thư viện sau khi tạo manifest.
- `python -m pytest`: chạy kiểm thử sau khi cài pytest.

Các lệnh trên chưa được kiểm chứng tại workspace.

## Giao diện báo cáo bằng Gradio

Dùng Gradio để tải ảnh và hiển thị kết quả SVM, CNN và fusion cạnh nhau. Mỗi phương pháp hiển thị nhãn độ chín, tươi/héo và xác suất dự đoán. Hiển thị ảnh gốc cùng vùng quả đã xử lý để giải thích pipeline. Bổ sung bảng chỉ số và confusion matrix từ đánh giá test đã lưu; không tính chỉ số toàn bộ mô hình từ một ảnh demo. Giao diện chỉ nạp mô hình đã huấn luyện, không huấn luyện lại khi dự đoán. Giữ logic xử lý trong `src/`, tách khỏi giao diện. Hướng dẫn xây dựng Gradio từng bước sau phần mô hình và fusion.

## Coding Style & Naming Conventions

Dùng Python, thụt lề bốn dấu cách; `snake_case` cho hàm, biến và file. Viết hàm ngắn, chú thích tiếng Việt. Chưa cấu hình formatter hoặc linter.

## Testing & Experiment Guidelines

Dự kiến dùng pytest, đặt tên `test_*.py`; chưa có ngưỡng coverage. Kiểm tra kích thước feature, thứ tự lớp và fusion. Chia dữ liệu theo mã quả; ảnh cùng quả không được xuất hiện ở nhiều tập. Chỉ dùng train để fit scaler, validation để chọn cấu hình, test để đánh giá cuối. Báo cáo Macro-F1 từng thuộc tính và confusion matrix; không bịa kết quả.

## Commit & Pull Request Guidelines

Chưa có Git history để suy ra quy ước. Khi khởi tạo Git, dùng commit ngắn như `docs: explain HSV histogram`. PR mô tả thay đổi, kiến thức liên quan và cách kiểm tra; đính kèm biểu đồ khi phù hợp. Không commit môi trường ảo, dữ liệu lớn hoặc model checkpoint.
