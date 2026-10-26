# Repository Guidelines

## Mục tiêu và cách cộng tác

Đồ án phân loại mỗi quả tắc theo hai thuộc tính độc lập: độ chín (`xanh/chín`) và độ tươi (`tươi/héo`). Không gộp hai thuộc tính thành một nhãn duy nhất. Hai thành viên phụ trách nhánh HSV + LBP + SVM và nhánh CNN so sánh MobileNetV2 với ResNet50; mỗi phương pháp dự đoán cả hai thuộc tính, sau đó fusion xác suất.

Hướng dẫn lý thuyết trước, code sau. Giải thích đầu vào, đầu ra và lý do lựa chọn. Chia thành bài nhỏ, đưa ví dụ tối thiểu rồi để người học thực hành. Giải thích nguyên nhân khi sửa lỗi. Không hoàn thành đồ án thay người học; chỉ triển khai phần được yêu cầu.

## Thiết kế mô hình và fusion

- **SVM:** dùng histogram HSV mô tả màu sắc và histogram LBP mô tả kết cấu vỏ. Ghép hai histogram thành vector, chuẩn hóa rồi huấn luyện hai SVM riêng cho độ chín và độ tươi. Mỗi SVM xuất nhãn và xác suất.
- **LBP:** cấu hình khởi đầu `P=8`, `R=1`, `method="uniform"`; histogram 10 bin chuẩn hóa tổng bằng 1. Tính trên ảnh xám, chỉ thống kê tại điểm có toàn bộ vùng lân cận nằm trong mask quả. Báo lỗi nếu không có điểm hợp lệ.
- **CNN:** so sánh MobileNetV2 và ResNet50 bằng transfer learning từ ImageNet, dự kiến huấn luyện trên Kaggle. Mỗi mô hình dùng backbone chung, Global Average Pooling và hai đầu phân loại; mỗi đầu xuất xác suất cho hai lớp của một thuộc tính.
- **Fusion:** chọn CNN theo trung bình Macro-F1 validation của hai thuộc tính; nếu bằng nhau, ưu tiên MobileNetV2. Fusion xác suất SVM với CNN đã chọn cho từng thuộc tính; chọn trọng số trên validation. Giữ thống nhất thứ tự lớp giữa các mô hình.

## Quy ước ảnh đầu vào

- Mỗi ảnh chứa một quả tắc đã xóa nền; giữ ảnh gốc, mask và mã quả.
- Crop đủ toàn bộ quả, giữ hình dáng và kết cấu vỏ; không làm mịn hoặc kéo méo.
- Ghép nền RGB cố định, mặc định màu đen; giữ tỉ lệ và thêm padding để tạo ảnh `224 × 224`.
- HSV chỉ lấy pixel thuộc mask; LBP loại điểm có vùng lân cận chạm nền. CNN dùng ảnh RGB với tiền xử lý đúng theo backbone và trọng số pretrained.
- Áp dụng cùng quy trình cho train, validation, test và demo. Giao diện yêu cầu ảnh một quả đã tách nền; chưa bổ sung phát hiện nhiều quả hoặc tự động tách nền.

## Dữ liệu và gán nhãn

Dữ liệu hiện có nhưng chưa gán nhãn. Trước huấn luyện, thống nhất tiêu chí xanh/chín, tươi/héo và cách xử lý trường hợp khó xác định. Lưu mã quả để nhóm các ảnh cùng quả, kể cả ảnh chụp khác ngày.

## Project Structure & Module Organization

Repository hiện chưa có mã nguồn. Cấu trúc dự kiến khi cần triển khai:

- `src/`: tiền xử lý, đặc trưng HSV và LBP, SVM, CNN và fusion.
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

Dùng Gradio để tải ảnh một quả tắc đã tách nền và hiển thị kết quả SVM, CNN đã chọn và fusion cạnh nhau. Mỗi phương pháp hiển thị nhãn độ chín, tươi/héo và xác suất dự đoán. Hiển thị ảnh tải lên cùng vùng quả đã xử lý để giải thích pipeline. Bổ sung bảng chỉ số và confusion matrix từ đánh giá test đã lưu; không tính chỉ số toàn bộ mô hình từ một ảnh demo. Giao diện chỉ nạp mô hình đã huấn luyện, không huấn luyện lại khi dự đoán. Giữ logic xử lý trong `src/`, tách khỏi giao diện. Hướng dẫn xây dựng Gradio từng bước sau phần mô hình và fusion.

## Coding Style & Naming Conventions

Dùng Python, thụt lề bốn dấu cách; `snake_case` cho hàm, biến và file. Viết hàm ngắn, chú thích tiếng Việt. Chưa cấu hình formatter hoặc linter.

Code đơn giản không cầu kì viết tối thiểu.

## Testing & Experiment Guidelines

Dự kiến dùng pytest, đặt tên `test_*.py`; chưa có ngưỡng coverage. Kiểm tra kích thước feature, histogram LBP, xử lý mask không có điểm hợp lệ, thứ tự lớp và fusion.

Chia dữ liệu theo mã quả; ảnh cùng quả, kể cả chụp khác ngày, phải thuộc cùng một tập. So sánh `HSV + SVM` với `HSV + LBP + SVM`, đồng thời so sánh MobileNetV2 với ResNet50 trên cùng tập chia. Chỉ dùng train để fit scaler, validation để chọn cấu hình và trọng số fusion, test để đánh giá cuối. Báo cáo Macro-F1 từng thuộc tính và confusion matrix; không bịa kết quả hoặc khẳng định mô hình tốt hơn khi chưa có kết quả.

## Commit & Pull Request Guidelines

Chưa có Git history để suy ra quy ước. Khi khởi tạo Git, dùng commit ngắn như `docs: explain HSV histogram`. PR mô tả thay đổi, kiến thức liên quan và cách kiểm tra; đính kèm biểu đồ khi phù hợp. Không commit môi trường ảo, dữ liệu lớn hoặc model checkpoint.
