# Repository Guidelines

## Mục tiêu và cách cộng tác

Đồ án phân loại bề mặt đường từ ảnh thành **một trong ba lớp**: đường đất hoặc chưa lát, bê tông và đường lát đá khối. So sánh hai hướng: đặc trưng thủ công + SVM và CNN. Nhánh CNN gồm MobileNetV2 pretrained trên ImageNet và một CNN nhỏ tự xây dựng, huấn luyện từ đầu.

Đầu vào là ảnh đường phố; đầu ra là nhãn bề mặt tại vùng đường được khảo sát. Phạm vi hiện tại là phân loại ảnh, chưa triển khai phát hiện đường, phân đoạn, đánh giá chất lượng mặt đường hoặc fusion.

Hướng dẫn lý thuyết trước, code sau. Giải thích đầu vào, đầu ra và lý do lựa chọn. Chia thành bài nhỏ, đưa ví dụ tối thiểu rồi để người học thực hành. Giải thích nguyên nhân khi sửa lỗi. Không hoàn thành đồ án thay người học; chỉ triển khai phần được yêu cầu.

Hai thành viên dự kiến phụ trách: một người khảo sát HOG, RGB histogram, LBP và SVM; một người phụ trách MobileNetV2 và CNN tự xây dựng. Cùng thống nhất dữ liệu, tập chia, tiền xử lý và cách đánh giá trước khi huấn luyện.

## Dataset và nhãn

Nguồn chính là [StreetSurfaceVis V1.0 trên Zenodo](https://zenodo.org/records/11449977). Dùng nhãn `surface_type` trong `streetSurfaceVis_v1_0.csv`.

| ID quy ước | Nhãn gốc | Tên hiển thị | Số ảnh theo bảng công bố |
| --- | --- | --- | ---: |
| 0 | `unpaved` | Đường đất | 1.016 |
| 1 | `concrete` | Bê tông | 972 |
| 2 | `sett` | Đường lát đá khối | 1.363 |

Không dùng `surface_quality` làm mục tiêu.

Giữ `mapillary_image_id`, tọa độ, thời gian chụp và cờ `train`. Tác giả dành `train=False` cho test từ năm thành phố riêng. Ảnh có giấy phép CC-BY-SA; ghi nguồn, tác giả và giấy phép khi sử dụng, minh họa hoặc phân phối lại.

Trước khi huấn luyện, đối chiếu số lượng thực tế từ CSV, phân bố lớp theo từng tập, ảnh thiếu/hỏng và ảnh trùng. Số công bố ở trên không phải số lượng train/validation/test sau làm sạch.

### Dữ liệu bổ sung từ Deep Pavements

- Dùng [Deep Pavements Dataset](https://github.com/kauevestena/deep_pavements_dataset) tại revision `0a523b4727a3e9f6240c7a6e00e195295b5d3e48`. Ánh xạ `ground → unpaved (0)`, `concrete → concrete (1)`, `sett → sett (2)`. Danh sách patch đã rà trực quan nằm trong `src/deep_pavements_selection.txt`.
- Deep Pavements cung cấp patch mặt đường đã cắt sẵn. Resize giữ tỉ lệ và center crop về `224 × 224`, bỏ qua crop vùng dưới dùng cho ảnh StreetSurfaceVis. Dùng hash sai phân để sàng lọc ảnh gần trùng với ảnh đã tiền xử lý. Nguồn này thiếu metadata vị trí và chuỗi cho các patch nên kiểm tra hash chỉ là bước sàng lọc.
- Đặt ba thư mục lớp trong `dataset/deep_pavements_dataset/dataset/` và chạy `python src/add_deep_pavements.py` sau khi đã có manifest và ảnh StreetSurfaceVis tiền xử lý. Script ghi patch vào `outputs/preprocessed_deep_pavements/` và tạo `outputs/manifest_split_deep_pavements.csv`. Hai cột `source` và `source_url` ghi nguồn và đường dẫn tới patch ở revision đã chốt. Truy nguyên và kiểm tra thông tin ghi công, giấy phép từng ảnh gốc trước khi phân phối các patch.
- Chỉ thêm Deep Pavements vào train. Giữ `outputs/manifest_split.csv` cho baseline. Với thí nghiệm bổ sung, chọn `manifest_split_deep_pavements.csv` trong cell tìm manifest của notebook và đưa thư mục patch đã xử lý vào Kaggle cùng dữ liệu gốc. So sánh Macro-F1 và F1 từng lớp trên cùng validation. Giữ test StreetSurfaceVis cho đánh giá cuối.

### Thí nghiệm bổ sung bê tông từ RSCD

- [Road Surface Classification Dataset (RSCD)](https://thu-rsxd.com/rscd/) cung cấp patch mặt đường từ camera gắn trên xe ở Bắc Kinh. Dùng các ảnh có nhãn bê tông khô, ướt và đọng nước trong train của RSCD. Theo trang dự án, dữ liệu dùng cho nghiên cứu phi thương mại theo CC BY-NC. Ghi nguồn và giữ điều kiện đó khi chia sẻ.
- Chạy `python src/add_rscd_concrete.py` để tải mẫu từ revision đã chốt, lấy cách nhau ít nhất 45 giây theo timestamp tên ảnh, rồi center crop về `224 × 224`. Script tạo `outputs/manifest_split_rscd_concrete.csv` từ baseline và lưu ảnh mới vào `outputs/preprocessed_rscd_concrete/`. Timestamp giúp giảm khung liên tiếp, không xác định được đoạn đường độc lập.
- So sánh cấu hình StreetSurfaceVis với StreetSurfaceVis + RSCD bằng cùng cách xáo trộn train. Chọn `manifest_split_rscd_concrete.csv` cho cấu hình chỉ bổ sung RSCD với test StreetSurfaceVis gốc. Để chạy baseline, đổi `manifest_name` về `manifest_split.csv` và chọn `run_dir` riêng. Dùng `DataLoader(shuffle=True)` cho train của cả hai CNN.

## Chia dữ liệu và tránh rò rỉ

- Cấu hình hiện tại chia lại toàn bộ 3.385 ảnh trong `outputs/manifest_split.csv` theo tỉ lệ 80% train, 10% validation và 10% test, gồm 3.351 ảnh StreetSurfaceVis và 34 ảnh Mapillary bổ sung. Cờ `train` của StreetSurfaceVis mô tả cách chia gốc của tác giả. Kết quả theo cách chia mới cần được báo cáo với tên tập chia tương ứng.
- Chạy `python src/split_data.py` để cập nhật cột `split` ngay trong manifest chung. Script giữ nhóm hiện có, dùng `StratifiedGroupKFold` với 10 fold và seed 42, rồi chọn hai fold cho validation và test dựa trên số ảnh, tỉ lệ lớp và tỉ lệ nguồn. Cả ba tập phải có đủ ba lớp. Notebook đọc manifest chung và chọn checkpoint theo Macro-F1 trên validation của cách chia mới.
- Khảo sát tọa độ để chọn nhóm theo khu vực/đoạn đường; kết hợp thời gian chụp khi phù hợp. Metadata công bố không đảm bảo có sequence ID: không tự giả định tồn tại trường này, không dùng riêng ID ảnh hoặc người chụp làm nhóm tuyến đường.
- Kiểm tra ảnh giống/gần trùng và các vị trí sát nhau ở ranh giới tập chia. Dùng cụm vị trí và khoảng đệm phù hợp sau khảo sát; ghi rõ cách chọn và giới hạn còn lại. Ảnh cùng chuỗi chụp nếu xác định được phải cùng tập.
- Nếu phát hiện bản sao của ảnh test trong phần train/validation, loại bản sao khỏi phần phát triển và ghi log; không dùng kết quả test để quyết định.
- Lưu manifest gồm ID ảnh, đường dẫn, nhãn, nhóm và split. Cố định seed ban đầu `42`. Mọi mô hình dùng cùng manifest trong từng cấu hình dữ liệu.
- Chỉ dùng train để fit scaler, tính class weight và cập nhật trọng số mô hình. Validation dùng chọn cấu hình/checkpoint; test chỉ dùng đánh giá cuối sau khi chốt thí nghiệm.

### Test bổ sung cho lớp bê tông

- Giữ test StreetSurfaceVis làm mốc ba lớp. Dùng riêng `outputs/manifest_test_roadsaw_concrete.csv` để kiểm tra khả năng nhận ra bê tông trên patch nhìn từ trên xuống của [RoadSaW](https://viscoda.com/index.php/en/downloads-en/roadsaw-dataset). RoadSaW do VISCODA GmbH phát hành theo CC BY-NC-SA 4.0.
- Dùng các thư mục `test/concrete_*` trong archive `RoadSaW-075_s.zip`. Các khung liên tiếp được lấy cách nhau ít nhất 60 frame trong mỗi chuỗi. Ghi chuỗi trong `group_id` và đường dẫn trong ZIP ở `archive_member`. Chạy `python src/prepare_roadsaw_test.py` với mẫu đã tải và chỉ mục `dataset/roadsaw_075_sample/index.csv` để tạo ảnh RGB `224 × 224` trong `outputs/preprocessed_roadsaw_test/` và sàng lọc ảnh gần trùng.
- Nạp checkpoint đã chọn trên validation. Tạo Dataset riêng từ manifest RoadSaW với `split="test"` và dùng đúng transform của checkpoint. Báo recall bê tông, số ảnh và số chuỗi quay. Các khung từ cùng chuỗi hoặc cùng đường thử có tương quan. Dùng Macro-F1 ba lớp trên test StreetSurfaceVis và báo kết quả RoadSaW riêng.

### Test mở rộng cân bằng ba lớp

- `outputs/manifest_split_rscd_concrete_balanced_test.csv` giữ nguyên train và validation của bản StreetSurfaceVis + RSCD. Phần test giữ 133 ảnh StreetSurfaceVis rồi thêm 69 ảnh bê tông và 53 ảnh lát đá khối từ RoadSaW để đạt 85 ảnh mỗi lớp.
- Ánh xạ `cobble` của RoadSaW sang `sett` dựa trên dạng lát đá khối thấy trong các patch đã rà. Chọn tối đa một ảnh mỗi chuỗi và sàng lọc ảnh gần trùng. Dùng `python src/prepare_balanced_test.py` với chỉ mục và danh sách ảnh loại trong `dataset/roadsaw_balanced_candidates/` để xuất ảnh vào `outputs/preprocessed_balanced_test/`.
- Báo kết quả tập cân bằng bên cạnh test StreetSurfaceVis gốc. Các patch RoadSaW là ảnh nhìn từ trên xuống trên đường thử và chỉ bổ sung cho hai lớp. Số ảnh mỗi lớp bằng nhau nhưng nguồn ảnh vẫn lệch theo lớp, nên điểm trên tập mở rộng phản ánh hỗn hợp nguồn này.

### Thí nghiệm bổ sung đường lát đá từ RoadSaW

- `python src/add_roadsaw_sett.py` dùng danh sách `src/roadsaw_sett_selection.txt` và chỉ mục ảnh đã tải tại `dataset/roadsaw_sett_train/index.csv`. Chọn 86 patch `cobble` từ 20 chuỗi thuộc split `train` gốc của RoadSaW, gồm ảnh còn thấy khe đá và ảnh có vệt nhòe. Các khung được lấy cách nhau ít nhất 60 frame trong từng chuỗi. Kiểm tra chuỗi và sàng lọc ảnh gần trùng với tập cân bằng và test RoadSaW bê tông. Các chuỗi vẫn có thể chung đường thử hoặc ngày quay.
- Script tạo `manifest_split_rscd_roadsaw_sett.csv` và thư mục `outputs/preprocessed_roadsaw_sett_train/`. Bản manifest trước khi thêm CtD nằm trong `dataset/outputs_archive/`. Vì lỗi trên tập 255 ảnh đã được dùng để định hướng bổ sung dữ liệu, dùng tập này để phân tích lỗi và cần tập giữ riêng mới nếu báo cáo khả năng tổng quát hóa độc lập trên RoadSaW.

### Bổ sung bê tông và đá lát từ RoadSaW CtD

- Chạy `python src/add_roadsaw_ctd.py` với `dataset/roadsaw_ctd1.zip`, `dataset/roadsaw_ctd2.zip` và `dataset/roadsaw_ctd3.zip`. Script bổ sung 100 patch bê tông và 14 patch lát đá vào train, để phần RoadSaW train có 100 ảnh mỗi lớp. Giữ ba lớp `unpaved`, `concrete`, `sett`.
- Dùng nhãn bề mặt theo chuỗi trong archive, lấy khung cách nhau ít nhất 30 frame và sàng lọc ảnh gần trùng với dữ liệu đã có. Validation và test giữ cùng ảnh với trước khi bổ sung CtD.
- Notebook dùng `outputs/manifest_split_rscd_roadsaw.csv` và thư mục checkpoint riêng. Thêm manifest cùng `outputs/preprocessed_roadsaw_ctd_train/` vào bản dữ liệu Kaggle đang dùng.

### Chọn checkpoint bằng validation hai nguồn

- `python src/add_roadsaw_validation.py` sử dụng các patch đã tải và chỉ mục `dataset/roadsaw_validation_075/index.csv` để thêm 20 ảnh bê tông và 20 ảnh đá lát từ mười chuỗi RoadSaW-075_s chưa dùng. Script sàng lọc ảnh gần trùng và lấy các khung cách nhau ít nhất 60 frame trong mỗi chuỗi. Giữ nguyên 2.972 ảnh train và 255 ảnh test của cấu hình CtD.
- Notebook dùng `outputs/manifest_split_rscd_roadsaw_validation.csv`. Chọn checkpoint và early stopping theo trung bình có trọng số bằng nhau giữa Macro-F1 ba lớp trên 644 ảnh StreetSurfaceVis validation và Macro-F1 hai lớp trên 40 ảnh RoadSaW validation. Dự đoán `unpaved` cho ảnh RoadSaW vẫn làm giảm điểm lớp thật. Báo hai điểm theo nguồn cùng với điểm Macro-F1 toàn tập; chỉ số trung bình hai nguồn dùng để chọn checkpoint.

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

Dùng PyTorch và TorchVision trên Kaggle cho CNN tự xây dựng và MobileNetV2.

- **MobileNetV2** dùng trọng số `MobileNet_V2_Weights.IMAGENET1K_V1`, ảnh RGB đã crop về `224 × 224`, adaptive average pooling `1 × 1` và đầu phân loại Dropout 0.3 + Linear trả về ba logit.
- Với MobileNetV2, dùng `ToTensor()` rồi chuẩn hóa bằng mean và std của trọng số ImageNet. Giữ cùng vùng crop cho các mô hình.
- Huấn luyện hai giai đoạn: đóng băng backbone để học đầu phân loại, sau đó mở các block cuối và fine-tune với learning rate nhỏ hơn. Khởi đầu Adam `1e-3` khi chỉ học head. Khi fine-tune, dùng `1e-5` cho phần backbone vừa mở và `1e-4` cho head. Giữ thống kê BatchNorm cố định trong cả hai giai đoạn.
- **CNN tự xây dựng:** khởi tạo ngẫu nhiên, ba block Conv 3×3 + ReLU + MaxPool với 32, 64, 128 filter, rồi adaptive average pooling `2 × 2` → Dropout 0.3 → Linear trả về ba logit. Chuẩn hóa RGB về [0,1], không dùng tiền xử lý ImageNet.
- Quy ước nhãn nguyên 0–2 và `nn.CrossEntropyLoss()` trên logit. Bắt đầu batch size 32, giảm nếu thiếu bộ nhớ và ghi lại. Nếu dùng class weight, chỉ tính từ train và áp dụng cùng chính sách cho cả hai CNN.
- Đề xuất tối đa 10 epoch học head + 20 epoch fine-tune cho MobileNetV2; CNN tự xây dựng tối đa 30 epoch. Đây là ngân sách ban đầu, cần ghi số epoch và thời gian thực tế.
- Lưu checkpoint theo điểm validation, early stopping với patience khởi đầu 5 và khôi phục checkpoint tốt nhất. Dùng Macro-F1 ba lớp trên toàn bộ validation cho cấu hình StreetSurfaceVis và điểm trung bình hai nguồn cho cấu hình có RoadSaW validation.
- Dùng cùng tập chia, augmentation và tiêu chí chọn mô hình. Ghi rõ CNN tự xây dựng học từ đầu và MobileNetV2 dùng trọng số pretrained; không quy mọi chênh lệch chỉ cho kiến trúc.
- Khi đủ tài nguyên, chạy ba seed `42, 43, 44` cho cấu hình CNN đã chốt, báo trung bình và độ lệch chuẩn; không chọn seed dựa trên test.

## Kế hoạch triển khai từng bước

| Bước | Nội dung học và thực hành | Đầu ra để kiểm tra trước khi đi tiếp |
| --- | --- | --- |
| 1. Khảo sát dữ liệu | Đọc nhãn, xem ảnh từng lớp, phân biệt paving_stones và sett, kiểm tra lệch lớp/ảnh lỗi | Bảng thống kê, ảnh minh họa, danh sách ảnh loại và lý do |
| 2. Chốt dữ liệu | Tạo tập chia có nhóm, kiểm tra trùng lặp, thử crop trên vài ảnh trước khi xử lý hàng loạt | Manifest cố định, thống kê ba tập, ảnh crop kiểm tra trực quan |
| 3. SVM | Học histogram, HOG, LBP và chuẩn hóa; trích feature một ảnh rồi mới chạy S1–S5 | Vector đúng kích thước, pipeline SVM đã lưu, bảng kết quả validation |
| 4. CNN | Học convolution, transfer learning và fine-tuning; thử một batch rồi huấn luyện hai CNN | Kiểm tra đầu ra ba lớp, learning curves, checkpoint và log validation |
| 5. Đánh giá | Chốt cấu hình bằng validation, đánh giá các mô hình đã định trước trên test | Bảng so sánh, confusion matrix, phân tích lỗi theo lớp |
| 6. Demo và báo cáo | Tải checkpoint có sẵn, giải thích crop/feature và kết quả dự đoán | Gradio, hướng dẫn chạy, báo cáo có nguồn và giới hạn |

Nhánh SVM và CNN có thể làm song song sau bước 2. Không tạo sẵn module hay triển khai huấn luyện/giao diện khi chưa học đến bước tương ứng.

## Đánh giá và tái lập thí nghiệm

- Chỉ số chính: **Macro-F1 ba lớp**, vì số lượng lớp không cân bằng. Kèm accuracy, precision/recall/F1 từng lớp, support và confusion matrix 3×3 với thứ tự `[unpaved, concrete, sett]`.
- Thêm mốc majority-class, xác định lớp phổ biến từ train. So sánh S1–S5 và hai CNN trên cùng dữ liệu đánh giá.
- Chọn cấu hình mỗi phương pháp bằng điểm validation đã định cho thí nghiệm. Nếu bằng nhau, ưu tiên mô hình đơn giản/nhẹ hơn. Cố định cấu hình trước khi chạy test; không dùng test để chọn hyperparameter hoặc quay lại chỉnh mô hình.
- Ghi thời gian huấn luyện, thời gian suy luận sau warm-up, kích thước artifact và số tham số CNN. Đo cùng phần cứng/batch size, nêu rõ thời gian có bao gồm crop và trích feature không.
- Lưu cấu hình, seed, manifest, phiên bản thư viện, lịch sử huấn luyện, dự đoán và chỉ số. Không bịa kết quả, không khẳng định mô hình tốt hơn khi chưa có thực nghiệm.
- Phân tích nhầm lẫn giữa các lớp, bóng râm, ẩm ướt và vùng có nhiều vật liệu bằng ví dụ thực tế. Demo chỉ xử lý ba lớp đã học; ảnh ngoài miền không có bảo đảm phân loại đúng.

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

Tải ảnh đường phố, hiển thị ảnh gốc và vùng crop thực sự đưa vào mô hình. So sánh SVM đã chọn, MobileNetV2 và CNN tự xây dựng; mỗi mô hình hiển thị nhãn và xác suất ba lớp nếu có xác suất hợp lệ.

Chỉ nạp mô hình đã huấn luyện, không huấn luyện lại khi dự đoán. Giữ logic trong `src/`, tách khỏi giao diện. Hiển thị bảng chỉ số và confusion matrix từ đánh giá test đã lưu; không tính chỉ số toàn bộ mô hình từ một ảnh demo. Nêu rõ nhãn áp dụng cho vùng crop và ba lớp trong phạm vi đề tài.

## Coding Style & Testing Guidelines

Dùng Python, thụt lề bốn dấu cách; `snake_case` cho hàm, biến và file. Viết hàm ngắn, chú thích tiếng Anh. Code đơn giản, tối thiểu; chưa cấu hình formatter hoặc linter.

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
- [TorchVision MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html) mô tả trọng số ImageNet và cách chuẩn hóa ảnh.
