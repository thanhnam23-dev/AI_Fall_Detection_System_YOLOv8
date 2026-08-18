import cv2
import numpy as np
import os
import glob
import pandas as pd
import urllib.request
from tqdm import tqdm
from ultralytics import YOLO

# 1. Khởi tạo mô hình
YOLO_PATH = "models/yolov8n-pose.pt" if os.path.exists("models/yolov8n-pose.pt") else "yolov8n-pose.pt"
model = YOLO(YOLO_PATH)

# Đường dẫn thư mục dữ liệu
DATA_DIR = "S:/AI_Fall_Detection_System_YOLOv8/data"
FALL_DIR = os.path.join(DATA_DIR, "Fall")
ADL_DIR = os.path.join(DATA_DIR, "ADL")
OUTPUT_CSV = "S:/AI_Fall_Detection_System_YOLOv8/urfd_dataset.csv"

# Tạo thư mục tạm để chứa các file đồng bộ CSV từ URFD
CSV_TEMP_DIR = os.path.join(DATA_DIR, "csv_temp")
os.makedirs(CSV_TEMP_DIR, exist_ok=True)

# Hàm tải file đồng bộ từ URFD để biết frame ngã chính xác
def download_urfd_csv(seq_num):
    url = f"https://fenix.ur.edu.pl/~mkepski/ds/data/fall-{seq_num:02d}-data.csv"
    dest_path = os.path.join(CSV_TEMP_DIR, f"fall-{seq_num:02d}-data.csv")
    if not os.path.exists(dest_path):
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception as e:
            print(f"Không thể tải file CSV cho fall-{seq_num:02d}: {e}")
            return None
    return dest_path

# Hàm tìm frame ngã dựa trên gia tốc kế (SV_total lớn nhất hoặc vượt ngưỡng)
def get_fall_impact_frame(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, header=None)
        # Cột 0: Frame, Cột 2: SV_total
        # Tìm frame có gia tốc lớn nhất (thời điểm va chạm)
        max_idx = df[2].idxmax()
        impact_frame = int(df.iloc[max_idx][0])
        return impact_frame
    except Exception as e:
        print(f"Lỗi đọc file CSV: {e}")
        return None

# Hàm tính toán góc nghiêng thân người giống detect.py
def calculate_angle(keypoints):
    def get_pt_conf(idx):
        if idx >= len(keypoints):
            return np.array([0.0, 0.0]), 0.0
        pt = keypoints[idx][:2]
        conf = keypoints[idx][2] if len(keypoints[idx]) > 2 else 1.0
        return pt, conf

    nose, nose_conf = get_pt_conf(0)
    l_sh, l_sh_conf = get_pt_conf(5)
    r_sh, r_sh_conf = get_pt_conf(6)
    l_hip, l_hip_conf = get_pt_conf(11)
    r_hip, r_hip_conf = get_pt_conf(12)

    if l_sh_conf > 0.3 and r_sh_conf > 0.3:
        sh_mid = (l_sh + r_sh) / 2.0
    elif l_sh_conf > 0.3:
        sh_mid = l_sh
    elif r_sh_conf > 0.3:
        sh_mid = r_sh
    else:
        sh_mid = nose

    if l_hip_conf > 0.3 and r_hip_conf > 0.3:
        hip_mid = (l_hip + r_hip) / 2.0
    elif l_hip_conf > 0.3:
        hip_mid = l_hip
    elif r_hip_conf > 0.3:
        hip_mid = r_hip
    else:
        hip_mid = nose

    u = sh_mid - hip_mid
    v = np.array([0, -1])
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u > 0 and norm_v > 0:
        cos_theta = np.dot(u, v) / (norm_u * norm_v)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle_rad)
    return 0.0

# Thu thập dữ liệu đặc trưng
dataset = []

# ==================== PHẦN 1: XỬ LÝ VIDEO FALL ====================
print("--- Đang xử lý các video ngã (Fall) ---")
fall_videos = glob.glob(os.path.join(FALL_DIR, "fall-*-cam0*.mp4"))

for video_path in tqdm(fall_videos):
    # Trích xuất số thứ tự video từ tên file (ví dụ: fall-01 -> 1)
    filename = os.path.basename(video_path)
    parts = filename.split('-')
    try:
        seq_num = int(parts[1])
    except ValueError:
        continue
    
    # Tải file đồng bộ và tìm frame va chạm
    csv_path = download_urfd_csv(seq_num)
    impact_frame = get_fall_impact_frame(csv_path)
    
    # Nếu không tìm được frame va chạm từ CSV, dùng mặc định là nửa video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not impact_frame:
        impact_frame = total_frames // 2
        
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Nhận diện pose (giữ conf thấp để lấy được box khi nằm sàn)
        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        
        # Gán nhãn cho frame hiện tại
        # Nhãn 0: Bình thường (trước khi ngã)
        # Nhãn 1: Đang ngã & nằm sàn (từ lúc chuẩn bị ngã đến hết)
        label = 1 if frame_idx >= (impact_frame - 10) else 0
        
        # Chỉ lấy người có độ tin cậy tốt nhất (YOLO thường trả về nhiều box ảo)
        if len(boxes) > 0 and len(keypoints_data) > 0:
            # Chọn người có diện tích box lớn nhất (đối tượng chính)
            areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
            best_idx = np.argmax(areas)
            
            box = boxes[best_idx]
            keypoints = keypoints_data[best_idx]
            
            # Chuẩn hóa tọa độ 17 khớp xương về khoảng [0, 1] relative với bounding box
            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            
            normalized_kpts = []
            for kp in keypoints:
                kx, ky = kp[0], kp[1]
                # Chuẩn hóa
                kx_norm = (kx - x1) / w
                ky_norm = (ky - y1) / h
                normalized_kpts.extend([kx_norm, ky_norm])
            
            # Đặc trưng hình học bổ trợ
            angle = calculate_angle(keypoints)
            aspect_ratio = w / h
            
            # Lưu đặc trưng vào dataset
            row = normalized_kpts + [angle, aspect_ratio, label]
            dataset.append(row)
            
    cap.release()

# ==================== PHẦN 2: XỬ LÝ VIDEO ADL ====================
print("\n--- Đang xử lý các video hoạt động thường ngày (ADL) ---")
adl_videos = glob.glob(os.path.join(ADL_DIR, "adl-*-cam0*.mp4"))

for video_path in tqdm(adl_videos):
    cap = cv2.VideoCapture(video_path)
    
    # Đối với ADL, tất cả các frame đều là trạng thái Bình thường (Nhãn 0)
    label = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        
        if len(boxes) > 0 and len(keypoints_data) > 0:
            areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
            best_idx = np.argmax(areas)
            
            box = boxes[best_idx]
            keypoints = keypoints_data[best_idx]
            
            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            
            normalized_kpts = []
            for kp in keypoints:
                kx, ky = kp[0], kp[1]
                kx_norm = (kx - x1) / w
                ky_norm = (ky - y1) / h
                normalized_kpts.extend([kx_norm, ky_norm])
            
            angle = calculate_angle(keypoints)
            aspect_ratio = w / h
            
            row = normalized_kpts + [angle, aspect_ratio, label]
            dataset.append(row)
            
    cap.release()

# ==================== PHẦN 3: GHI RA FILE CSV ====================
# Đặt tên các cột: 17 khớp x 2 tọa độ (x, y) = 34 cột + angle + aspect_ratio + label
columns = []
for i in range(17):
    columns.extend([f"kp_{i}_x", f"kp_{i}_y"])
columns.extend(["angle", "aspect_ratio", "label"])

df_dataset = pd.DataFrame(dataset, columns=columns)
# Loại bỏ các dòng bị thiếu dữ liệu (nếu có)
df_dataset.dropna(inplace=True)
df_dataset.to_csv(OUTPUT_CSV, index=False)

print(f"\n Hoàn thành trích xuất! File dữ liệu lưu tại: '{OUTPUT_CSV}'")
print(f"Tổng số mẫu thu thập được: {len(df_dataset)}")
print(f"Số mẫu bình thường (0): {len(df_dataset[df_dataset['label'] == 0])}")
print(f"Số mẫu ngã (1): {len(df_dataset[df_dataset['label'] == 1])}")
