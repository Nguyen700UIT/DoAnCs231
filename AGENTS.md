# Repository Guidelines

## Mục tiêu và cách cộng tác

Đồ án phân loại bề mặt đường từ ảnh thành **một trong ba lớp**: nhựa đường, bê tông và đường lát gạch. So sánh hai hướng: đặc trưng thủ công + SVM và CNN. Nhánh CNN gồm MobileNetV2, ResNet-50 pretrained trên ImageNet và một CNN nhỏ tự xây dựng, huấn luyện từ đầu.

Đầu vào là ảnh đường phố; đầu ra là nhãn bề mặt tại vùng đường được khảo sát. Phạm vi hiện tại là phân loại ảnh, chưa triển khai phát hiện đường, phân đoạn, đánh giá chất lượng mặt đường hoặc fusion.

Hướng dẫn lý thuyết trước, code sau. Giải thích đầu vào, đầu ra và lý do lựa chọn. Chia thành bài nhỏ, đưa ví dụ tối thiểu rồi để người học thực hành. Giải thích nguyên nhân khi sửa lỗi. Không hoàn thành đồ án thay người học; chỉ triển khai phần được yêu cầu.

Hai thành viên dự kiến phụ trách: một người khảo sát HOG, RGB histogram, LBP và SVM; một người phụ trách MobileNetV2, ResNet-50 và CNN tự xây dựng. Cùng thống nhất dữ liệu, tập chia, tiền xử lý và cách đánh giá trước khi huấn luyện.

## Dataset và nhãn

Nguồn: [StreetSurfaceVis V1.0 trên Zenodo](https://zenodo.org/records/11449977). Dùng nhãn `surface_type` trong `streetSurfaceVis_v1_0.csv`.

| ID quy ước | Nhãn gốc | Tên hiển thị | Số ảnh theo bảng công bố |
| --- | --- | --- | ---: |
| 0 | `asphalt` | Nhựa đường | 3.735 |
| 1 | `concrete` | Bê tông | 972 |
| 2 | `paving_stones` | Đường lát gạch | 2.037 |

Tổng dự kiến: **6.744 ảnh** trước kiểm tra file và trùng lặp. Loại `sett`, `unpaved`; không gộp `sett` vào `paving_stones`. Không dùng `surface_quality` làm mục tiêu.

Giữ `mapillary_image_id`, tọa độ, thời gian chụp và cờ `train`. Tác giả dành `train=False` cho test từ năm thành phố riêng. Ảnh có giấy phép CC-BY-SA; ghi nguồn, tác giả và giấy phép khi sử dụng, minh họa hoặc phân phối lại.

Trước khi huấn luyện, đối chiếu số lượng thực tế từ CSV, phân bố lớp theo từng tập, ảnh thiếu/hỏng và ảnh trùng. Số công bố ở trên không phải số lượng train/validation/test sau làm sạch.

## Chia dữ liệu và tránh rò rỉ

- Sau khi lọc ba lớp, giữ tập test theo `train=False` của dataset; không chia ngẫu nhiên lại toàn bộ ảnh.
- Tách validation từ phần `train=True`, mục tiêu khoảng 20% phần này, bằng nhóm không gian; kiểm tra cả ba lớp đều có mặt trong train và validation. Tỉ lệ có thể lệch để giữ nguyên nhóm.
- Khảo sát tọa độ để chọn nhóm theo khu vực/đoạn đường; kết hợp thời gian chụp khi phù hợp. Metadata công bố không đảm bảo có sequence ID: không tự giả định tồn tại trường này, không dùng riêng ID ảnh hoặc người chụp làm nhóm tuyến đường.
- Kiểm tra ảnh giống/gần trùng và các vị trí sát nhau ở ranh giới tập chia. Dùng cụm vị trí và khoảng đệm phù hợp sau khảo sát; ghi rõ cách chọn và giới hạn còn lại. Ảnh cùng chuỗi chụp nếu xác định được phải cùng tập.
- Nếu phát hiện bản sao của ảnh test trong phần train/validation, loại bản sao khỏi phần phát triển và ghi log; không dùng kết quả test để quyết định.
- Lưu một manifest chung gồm ID ảnh, đường dẫn, nhãn, nhóm và split. Cố định seed ban đầu `42`; mọi mô hình sử dụng cùng manifest.
- Chỉ dùng train để fit scaler, tính class weight và cập nhật trọng số mô hình. Validation dùng chọn cấu hình/checkpoint; test chỉ dùng đánh giá cuối sau khi chốt thí nghiệm.

## Quy ước ảnh đầu vào

Theo [khuyến nghị của dataset](https://zenodo.org/records/11449977), crop vùng có tọa độ `(0.25W, 0.5H, 0.75W, H)` từ ảnh gốc.

- Đề xuất dùng bản `s_1024.zip` cho thí nghiệm chính; bản 256 chỉ dùng kiểm tra pipeline nhanh. Không trộn các bản độ phân giải như các ảnh độc lập.
- Đọc ảnh RGB và xử lý hướng EXIF trước khi xác định crop; thống nhất cách làm tròn tọa độ. Giữ ảnh gốc và ID, không tách nền hoặc dùng mask quả.
- Crop vùng đường, resize giữ tỉ lệ để cạnh ngắn bằng 224 pixel, rồi center crop thành `224 × 224`, không thêm padding. Lưu thông số hình học để tái lập; kiểm tra trực quan vùng cắt cuối vẫn thể hiện bề mặt đường được gán nhãn.
- Dùng cùng vùng crop và quy tắc hình học cho mọi mô hình. SVM lấy HOG/LBP trên ảnh xám và histogram trên ảnh RGB; CNN nhận RGB với tiền xử lý riêng cho backbone.
- Không làm mịn mạnh vì có thể mất kết cấu. Chỉ augmentation trên train; bắt đầu với lật ngang, biến đổi sáng nhẹ và xoay góc nhỏ. Tránh biến đổi làm vùng ảnh chuyển sang bề mặt khác hoặc thay đổi màu quá mức.
- Validation, test và demo dùng quy trình xác định, không augmentation ngẫu nhiên. Không dùng metadata vị trí/người chụp làm đặc trưng dự đoán.

## Nhánh đặc trưng thủ công + SVM

Các cấu hình dưới đây là điểm khởi đầu, cần xác nhận bằng validation. HOG mô tả hướng biên; RGB histogram mô tả phân bố màu; LBP mô tả kết cấu cục bộ. Đây là giả thuyết để thí nghiệm, không khẳng định đặc trưng nào tốt hơn trước khi đo.

- **RGB histogram:** 32 bin mỗi kênh trên giá trị nguyên 0–255, chuẩn hóa từng histogram có tổng bằng 1 rồi nối thành vector 96 chiều. Không nhầm với histogram RGB 3D.
- **HOG:** ảnh xám, `orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, `block_norm="L2-Hys"`. Với ảnh `224 × 224`, vector dự kiến 6.084 chiều; ghép RGB thành 6.180 chiều.
- **LBP:** ảnh xám uint8, `P=8`, `R=1`, `method="uniform"`; histogram 10 bin, chuẩn hóa tổng bằng 1. Loại biên ngoài một pixel để vùng lân cận đầy đủ; báo lỗi nếu không còn điểm hợp lệ. LBP + RGB có 106 chiều.
- **SVM:** dùng pipeline ghép feature → `StandardScaler` → `SVC` đa lớp. Fit scaler trong train/fold huấn luyện, không fit trên toàn bộ dữ liệu.
- Thử nhỏ: kernel linear với `C ∈ {0.1, 1, 10}`; RBF với cùng các giá trị C và `gamma="scale"`. So sánh `class_weight=None` với `"balanced"` trên validation. Chưa thêm PCA hoặc tìm kiếm lớn.
- SVM chỉ cần nhãn/decision score cho đánh giá phân loại. Nếu demo cần xác suất, hiệu chỉnh bằng các fold có nhóm trong train; không coi decision score là xác suất, không hiệu chỉnh trên test.

Ma trận thí nghiệm tối thiểu:

| Mã | Đặc trưng | Bộ phân loại | Mục đích |
| --- | --- | --- | --- |
| S1 | RGB histogram | SVM | Mốc so sánh màu sắc |
| S2 | HOG | SVM | Mốc so sánh hướng biên |
| S3 | HOG + RGB histogram | SVM | Cấu hình chính |
| S4 | LBP | SVM | Mốc so sánh kết cấu |
| S5 | LBP + RGB histogram | SVM | Phương án thay HOG |

Không tự mở rộng sang HOG + LBP + RGB trước khi hoàn thành các thí nghiệm này.

## Nhánh CNN

Đề xuất dùng TensorFlow/Keras trên Kaggle cho cả ba CNN để thống nhất quy trình. Chỉ bổ sung dependency khi bắt đầu triển khai.

- **MobileNetV2 và ResNet-50:** dùng `weights="imagenet"`, `include_top=False`, đầu vào `224 × 224 × 3`; thêm Global Average Pooling, Dropout khởi đầu 0.3 và Dense 3 lớp softmax.
- Dùng đúng `preprocess_input` của từng backbone; không dùng chung phép chia 255 rồi tiếp tục gọi tiền xử lý pretrained. Tách tiền xử lý khỏi augmentation và kiểm tra không áp dụng hai lần.
- Huấn luyện hai giai đoạn: đóng băng backbone để học đầu phân loại, sau đó mở các block cuối và fine-tune với learning rate nhỏ hơn. Khởi đầu Adam `1e-3` cho head, `1e-5` cho fine-tune; giữ BatchNormalization của backbone ở chế độ inference khi fine-tune.
- **CNN tự xây dựng:** khởi tạo ngẫu nhiên, ba block Conv 3×3 + ReLU + MaxPool với 32, 64, 128 filter, rồi Global Average Pooling → Dropout 0.3 → Dense 3 softmax. Chuẩn hóa RGB về [0,1], không dùng tiền xử lý ImageNet.
- Quy ước nhãn nguyên 0–2 và sparse categorical cross-entropy. Bắt đầu batch size 32, giảm nếu thiếu bộ nhớ và ghi lại. Nếu dùng class weight, chỉ tính từ train và áp dụng cùng chính sách cho ba CNN.
- Đề xuất tối đa 10 epoch học head + 20 epoch fine-tune cho mỗi backbone; CNN tự xây dựng tối đa 30 epoch. Đây là ngân sách ban đầu, cần ghi số epoch và thời gian thực tế.
- Lưu checkpoint có Macro-F1 validation cao nhất, early stopping với patience khởi đầu 5 và khôi phục checkpoint tốt nhất. Macro-F1 phải được tính trên toàn bộ validation với đủ ba nhãn.
- Dùng cùng tập chia, augmentation và tiêu chí chọn mô hình. Ghi rõ CNN tự xây dựng học từ đầu còn hai backbone có lợi thế pretraining; không quy mọi chênh lệch chỉ cho kiến trúc.
- Khi đủ tài nguyên, chạy ba seed `42, 43, 44` cho cấu hình CNN đã chốt, báo trung bình và độ lệch chuẩn; không chọn seed dựa trên test.

## Kế hoạch triển khai từng bước

| Bước | Nội dung học và thực hành | Đầu ra để kiểm tra trước khi đi tiếp |
| --- | --- | --- |
| 1. Khảo sát dữ liệu | Đọc nhãn, xem ảnh từng lớp, phân biệt paving_stones và sett, kiểm tra lệch lớp/ảnh lỗi | Bảng thống kê, ảnh minh họa, danh sách ảnh loại và lý do |
| 2. Chốt dữ liệu | Tạo tập chia có nhóm, kiểm tra trùng lặp, thử crop trên vài ảnh trước khi xử lý hàng loạt | Manifest cố định, thống kê ba tập, ảnh crop kiểm tra trực quan |
| 3. SVM | Học histogram, HOG, LBP và chuẩn hóa; trích feature một ảnh rồi mới chạy S1–S5 | Vector đúng kích thước, pipeline SVM đã lưu, bảng kết quả validation |
| 4. CNN | Học convolution, transfer learning và fine-tuning; thử một batch rồi huấn luyện ba CNN | Kiểm tra đầu ra ba lớp, learning curves, checkpoint và log validation |
| 5. Đánh giá | Chốt cấu hình bằng validation, đánh giá các mô hình đã định trước trên test | Bảng so sánh, confusion matrix, phân tích lỗi theo lớp |
| 6. Demo và báo cáo | Tải checkpoint có sẵn, giải thích crop/feature và kết quả dự đoán | Gradio, hướng dẫn chạy, báo cáo có nguồn và giới hạn |

Nhánh SVM và CNN có thể làm song song sau bước 2. Không tạo sẵn module hay triển khai huấn luyện/giao diện khi chưa học đến bước tương ứng.

## Đánh giá và tái lập thí nghiệm

- Chỉ số chính: **Macro-F1 ba lớp**, vì số lượng lớp không cân bằng. Kèm accuracy, precision/recall/F1 từng lớp, support và confusion matrix 3×3 với thứ tự `[asphalt, concrete, paving_stones]`.
- Thêm mốc majority-class, xác định lớp phổ biến từ train. So sánh S1–S5 và ba CNN trên cùng dữ liệu đánh giá.
- Chọn cấu hình mỗi phương pháp bằng Macro-F1 validation; nếu bằng nhau, ưu tiên mô hình đơn giản/nhẹ hơn. Cố định cấu hình trước khi chạy test; không dùng test để chọn hyperparameter hoặc quay lại chỉnh mô hình.
- Ghi thời gian huấn luyện, thời gian suy luận sau warm-up, kích thước artifact và số tham số CNN. Đo cùng phần cứng/batch size, nêu rõ thời gian có bao gồm crop và trích feature không.
- Lưu cấu hình, seed, manifest, phiên bản thư viện, lịch sử huấn luyện, dự đoán và chỉ số. Không bịa kết quả, không khẳng định mô hình tốt hơn khi chưa có thực nghiệm.
- Phân tích nhầm nhựa đường/bê tông, bóng râm, ẩm ướt và vùng có nhiều vật liệu bằng ví dụ thực tế. Demo chỉ xử lý ba lớp đã học; ảnh ngoài miền không có bảo đảm phân loại đúng.

## Project Structure & Module Organization

Code xử lý và gán nhãn ảnh tắc cùng test/dependency riêng đã được gỡ khi đổi đề tài. Hiện chưa có code mô hình bề mặt đường hoặc dependency manifest cho đề tài mới.

Cấu trúc dự kiến khi đến bước triển khai:

- `src/`: dữ liệu, tiền xử lý, feature, SVM, CNN và đánh giá.
- `notebooks/`: thực hành, khảo sát và thí nghiệm Kaggle.
- `tests/`: kiểm thử logic dữ liệu, feature, tiền xử lý và đánh giá.
- `data/`: ảnh và metadata; `outputs/`: manifest, mô hình, log, biểu đồ và báo cáo.
- `app.py`: Gradio, chỉ tạo sau khi hoàn thành đánh giá.

Ảnh, output và môi trường cũ trên máy không được xem là dữ liệu hay kết quả của đề tài mới. Không tự xóa dữ liệu cục bộ khi chưa được yêu cầu.

## Build, Test, and Development Commands

Khi bắt đầu code, tạo manifest tối thiểu theo phần đang học. Các lệnh dự kiến:

- `python -m venv .venv`: tạo môi trường.
- `python -m pip install -r requirements.txt`: cài sau khi đã tạo manifest mới.
- `python -m pytest`: chạy sau khi có test và dependency tương ứng.

Chưa xác nhận môi trường hiện tại phù hợp với đề tài mới. Không dùng dependency cũ của rembg để suy ra môi trường huấn luyện CNN.

## Giao diện báo cáo bằng Gradio

Tải ảnh đường phố, hiển thị ảnh gốc và vùng crop thực sự đưa vào mô hình. So sánh SVM đã chọn, MobileNetV2, ResNet-50 và CNN tự xây dựng; mỗi mô hình hiển thị nhãn và xác suất ba lớp nếu có xác suất hợp lệ.

Chỉ nạp mô hình đã huấn luyện, không huấn luyện lại khi dự đoán. Giữ logic trong `src/`, tách khỏi giao diện. Hiển thị bảng chỉ số và confusion matrix từ đánh giá test đã lưu; không tính chỉ số toàn bộ mô hình từ một ảnh demo. Nêu rõ nhãn áp dụng cho vùng crop và ba lớp trong phạm vi đề tài.

## Coding Style & Testing Guidelines

Dùng Python, thụt lề bốn dấu cách; `snake_case` cho hàm, biến và file. Viết hàm ngắn, chú thích tiếng Việt. Code đơn giản, tối thiểu; chưa cấu hình formatter hoặc linter.

Dự kiến dùng pytest với `test_*.py`, chưa có ngưỡng coverage. Khi triển khai, kiểm tra:

- Ánh xạ ba lớp, lọc nhãn, ID duy nhất và các tập chia không giao nhau theo ID/nhóm.
- Tọa độ crop, RGB/thang xám, kích thước ảnh, xử lý file lỗi và tiền xử lý riêng từng backbone.
- Kích thước HOG/RGB/LBP, histogram chuẩn hóa, feature hữu hạn và trường hợp không có điểm LBP hợp lệ.
- Scaler chỉ fit train, đầu ra CNN có ba xác suất tổng gần 1, thứ tự lớp khớp mọi artifact.
- Macro-F1 và confusion matrix giữ đủ ba nhãn kể cả khi không có dự đoán cho một lớp.

## Commit & Pull Request Guidelines

Dùng commit ngắn, có tiền tố như `docs: plan road surface classification`. PR mô tả thay đổi, kiến thức liên quan và cách kiểm tra; đính kèm biểu đồ khi phù hợp. Không commit môi trường ảo, dữ liệu lớn hoặc model checkpoint. Chỉ commit phần thay đổi thuộc yêu cầu đang làm.

## Tài liệu tham khảo

- [Dataset, nhãn, tập test và khuyến nghị crop](https://zenodo.org/records/11449977).
- [Bài báo StreetSurfaceVis](https://doi.org/10.1038/s41597-024-04295-9), dùng làm nguồn trích dẫn trong báo cáo.
- [scikit-image: HOG và LBP](https://scikit-image.org/docs/stable/api/skimage.feature.html).
- [Keras Applications](https://keras.io/api/applications/), đối chiếu tiền xử lý của MobileNetV2 và ResNet50 khi triển khai.
