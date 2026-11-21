
# Demo phân loại mặt đường

Ứng dụng Gradio dùng hai mô hình đã huấn luyện trong `notebooks/cs231-cnn.ipynb`. Tải ảnh đường phố để xem vùng cắt 224 × 224 và điểm dự đoán cho ba lớp đường đất, bê tông và lát đá khối.

Chạy từ thư mục dự án với môi trường Python đã kích hoạt.

```powershell
python -m pip install gradio torch torchvision pillow
python app.py
```

Mở địa chỉ Gradio in trên terminal. Nếu checkpoint nằm ở chỗ khác, truyền thư mục đó qua `python app.py --run-dir "D:\duong-dan\checkpoint"`.
