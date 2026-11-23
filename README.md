
# Demo phân loại mặt đường

Ứng dụng Gradio dùng hai CNN trong `notebooks/cs231-cnn.ipynb` và SVM trong `notebooks/cs231-svm.ipynb`. Tải ảnh đường phố để xem vùng cắt và dự đoán cho ba lớp đường đất, bê tông và lát đá khối.

Đặt `hog_hist_svm_pipeline.joblib` tải từ `/kaggle/working/` vào `models/` cùng checkpoint CNN. CNN giữ ảnh đầu vào 224 × 224; SVM nhận cùng vùng crop được resize về 128 × 128 và trích HOG + HSV đúng notebook.

Chạy từ thư mục dự án với môi trường Python đã kích hoạt.

```powershell
python -m pip install gradio torch torchvision pillow numpy opencv-python-headless scikit-image scikit-learn joblib
python app.py
```

Mở địa chỉ Gradio in trên terminal. Nếu checkpoint nằm ở chỗ khác, truyền thư mục đó qua `python app.py --run-dir "D:\duong-dan\checkpoint"`.
