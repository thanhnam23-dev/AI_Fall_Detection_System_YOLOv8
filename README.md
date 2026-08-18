# 🚨 AI Fall Detection System using YOLOv8-pose & Machine Learning (SVM)

Hệ thống Phát hiện Người ngã Thời gian thực kết hợp giữa **YOLOv8-pose** (trích xuất 17 khớp xương cơ thể), mô hình **Học máy SVM (Support Vector Machine)** và các giải pháp bộ lọc chống báo giả nâng cao.

---

## 📁 Cấu trúc Thư mục Dự án (Organized Project Layout)

```text
S:/AI_Fall_Detection_System_YOLOv8/
├── models/                     # Chứa các trọng số mô hình AI & Học máy
│   ├── yolov8n-pose.pt         # Mô hình YOLOv8-pose gốc (trích xuất khớp xương)
│   ├── fall_classifier.pkl     # Mô hình phân loại ngã SVM (đạt độ chính xác 96.00%)
│   └── scaler.pkl              # Bộ chuẩn hóa đặc trưng (StandardScaler)
│
├── utils/                      # Thư viện & Công cụ phụ trợ
│   ├── convert_to_h264.py      # Công cụ nén video kết quả sang chuẩn HTML5 H.264
│   └── test_env.py             # Script kiểm tra môi trường Python, PyTorch & CUDA
│
├── data/                       # Bộ dữ liệu thô URFD (Fall, ADL & Sync CSV)
├── video_test/                 # Thư mục chứa video đầu vào để test
├── video_result/               # Thư mục chứa video đầu ra kết quả nhận diện
│
├── detect_ml_live.py           # 🚀 SCRIPT CHÍNH: Nhận diện trực tiếp qua Camera Laptop (Đa luồng + ML)
├── detect_ml.py                # Nhận diện ngã trên file Video qua mô hình ML
├── detect.py                   # Nhận diện ngã bằng quy tắc hình học 2D
│
├── prepare_dataset.py          # Script 1: Trích xuất 17 khớp xương từ URFD sang CSV
├── train_classifier.py         # Script 2: Huấn luyện mô hình SVM từ file CSV
├── urfd_dataset.csv            # Tập dữ liệu 10.380 mẫu đặc trưng đã trích xuất
│
├── requirements.txt            # Thư viện Python phụ thuộc
├── environment.yml             # Cấu hình môi trường Conda (yolo_fall)
├── UPGRADE_GUIDE.md            # Tài liệu hướng dẫn 4 bước nâng cấp đồ án
└── README.md                   # Hướng dẫn sử dụng & Kiến trúc hệ thống
```

---

## 🚀 Hướng dẫn Chạy Hệ thống (Quickstart)

### 1. Kích hoạt môi trường máy ảo
```bash
conda activate yolo_fall
```

### 2. Chạy nhận diện trực tiếp qua Camera Laptop (Khuyên dùng)
```bash
python detect_ml_live.py
```
* **Tính năng:** Đọc camera đa luồng thời gian thực (không độ trễ tích lũy), dự đoán bằng SVM, có bộ lọc chống báo sai khi đứng quá gần camera hoặc cúi nhặt đồ, hiển thị FPS thực tế. Nhấn phím `q` để thoát.

### 3. Chạy nhận diện trên file Video thử nghiệm
```bash
python detect_ml.py
```

### 4. Nén video kết quả sang H.264 để preview mượt trên IDE/Browser
```bash
python utils/convert_to_h264.py video_result/posees2_ml.mp4
```

---

## 🔄 Quy trình Huấn luyện Mô hình Học Máy (ML Pipeline)

Nếu bạn bổ sung thêm video mới và muốn huấn luyện lại mô hình SVM:

1. **Bước 1: Trích xuất đặc trưng khớp xương ra file CSV**
   ```bash
   python prepare_dataset.py
   ```
2. **Bước 2: Huấn luyện mô hình SVM mới**
   ```bash
   python train_classifier.py
   ```
   *Mô hình mới sẽ tự động được lưu vào thư mục `models/`.*

---

## 🛡️ Các Nâng cấp & Giải pháp Chống báo nhầm (Features & Safeguards)

1. **Nâng cấp 1 (Tính toán Trục nghiêng cơ thể):** Tính góc nghiêng giữa trung điểm vai và trung điểm hông so với trục thẳng đứng.
2. **Nâng cấp 2 (Temporal Validation & Centroid Tracker):** Đếm trễ 30 frames (1 giây) kết hợp thuật toán theo dõi tâm (Centroid) để triệt tiêu hiện tượng nhảy ID và chống báo động giả.
3. **Nâng cấp 3 (Multi-threading Webcam Stream):** Đọc camera ngầm trên thread riêng biệt, đảm bảo video hiển thị 30 FPS thời gian thực không bị lag.
4. **Giải pháp 1 (Lọc đứt khung hình khi đứng gần camera):** Tự động ép về `Standing` nếu đối tượng đứng quá sát mép dưới camera làm mất mốc hông/chân.
5. **Giải pháp 2 (Foot Grounding Check):** So sánh vị trí Y giữa hông và cổ chân/đầu gối để phân biệt tư thế khom lưng/cúi nhặt đồ (chân đứng trụ bên dưới) với tư thế ngã nằm sàn thực sự.
