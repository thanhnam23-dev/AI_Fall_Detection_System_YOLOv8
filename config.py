import os

# =============================================================================
# FILE CẤU HÌNH HỆ THỐNG - CONFIG.PY
# Tất cả các tham số, đường dẫn và cấu hình ứng dụng được quản lý tập trung tại đây.
# =============================================================================

# 1. Đường dẫn các mô hình AI (Đặt trong thư mục models/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-pose.pt") if os.path.exists(os.path.join(MODELS_DIR, "yolov8n-pose.pt")) else "yolov8n-pose.pt"
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "fall_classifier.pkl") if os.path.exists(os.path.join(MODELS_DIR, "fall_classifier.pkl")) else "fall_classifier.pkl"
SCALER_MODEL_PATH = os.path.join(MODELS_DIR, "scaler.pkl") if os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")) else "scaler.pkl"

# 2. Cấu hình Nguồn camera đầu vào
# - Mặc định src = 0 (Camera Laptop / Webcam)
# - Dành cho Camera Imou/Dahua IP: "rtsp://admin:SAFETY_CODE@192.168.1.XX:554/cam/realmonitor?channel=1&subtype=0"
CAMERA_SOURCE = 0

# Kích thước màn hình hiển thị
OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 600

# 3. Ngưỡng cài đặt AI & Chống báo sai
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.35

# Ngưỡng đếm khung hình chống báo giả (30 frames ~ 1 giây duy trì ngã)
FALL_FRAME_THRESHOLD = 30
MAX_LOST_FRAMES = 30
TRACKER_MAX_DISTANCE = 180

# 4. Cấu hình Cảnh báo Telegram Bot (Sắp triển khai)
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
TELEGRAM_COOLDOWN_SECONDS = 15

# 5. Cấu hình Cơ sở dữ liệu SQLite (Sắp triển khai)
DB_ENABLED = True
DB_PATH = os.path.join(BASE_DIR, "fall_events.db")
