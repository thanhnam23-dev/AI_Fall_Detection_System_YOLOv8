import cv2
import numpy as np
import os
import sys
import pickle
import threading
import queue
import time
import cvzone
from ultralytics import YOLO

# 1. Khởi chạy mô hình YOLO
YOLO_PATH = "models/yolov8n-pose.pt" if os.path.exists("models/yolov8n-pose.pt") else "yolov8n-pose.pt"
try:
    model = YOLO(YOLO_PATH)
except Exception as e:
    print(f"Error loading YOLO: {e}")
    sys.exit(1)

# 2. Tải mô hình Học máy SVM và Scaler
MODEL_PATH = "models/fall_classifier.pkl" if os.path.exists("models/fall_classifier.pkl") else "fall_classifier.pkl"
SCALER_PATH = "models/scaler.pkl" if os.path.exists("models/scaler.pkl") else "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Error: Không tìm thấy file mô hình SVM trong 'models/'. Vui lòng chạy train_classifier.py trước!")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    clf = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# --- LỚP ĐA LUỒNG ĐỌC CAMERA (NÂNG CẤP 3) ---
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Giới hạn queue chỉ chứa tối đa 3 frames để giải phóng bộ đệm, luôn giữ frame mới nhất
        self.q = queue.Queue(maxsize=3)
        self.stopped = False
        self.ret = False
        
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
        
    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break
            
            # Nếu hàng đợi đầy, đẩy bớt frame cũ ra để nhường chỗ cho frame thời gian thực mới nhất
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
            
    def read(self):
        if self.q.empty():
            return None
        return self.q.get()
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# Hàm tính góc nghiêng cơ thể
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

def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    color_left = (255, 255, 0)
    color_right = (0, 165, 255)
    color_center = (0, 255, 0)
    connections = [
        (0, 1, color_left), (0, 2, color_right), 
        (1, 3, color_left), (2, 4, color_right),
        (5, 6, color_center), (5, 11, color_center), 
        (6, 12, color_center), (11, 12, color_center),
        (5, 7, color_left), (7, 9, color_left), 
        (6, 8, color_right), (8, 10, color_right),
        (11, 13, color_left), (13, 15, color_left), 
        (12, 14, color_right), (14, 16, color_right)
    ]
    for pt1_idx, pt2_idx, color in connections:
        if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
            continue
        pt1, pt2 = keypoints[pt1_idx], keypoints[pt2_idx]
        conf1 = pt1[2] if len(pt1) > 2 else 1.0
        conf2 = pt2[2] if len(pt2) > 2 else 1.0
        if conf1 > confidence_threshold and conf2 > confidence_threshold:
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
            
    for pt in keypoints:
        conf = pt[2] if len(pt) > 2 else 1.0
        if conf > confidence_threshold:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

# Hàm phụ trợ bổ sung logic Aspect Ratio & Giải pháp chống báo sai (Giải pháp 1 + Giải pháp 2)
def check_fall_status_fallback(keypoints, bbox=None, frame_height=600):
    # Dùng SVM để dự đoán chính
    # Trích xuất đặc trưng
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
    l_knee, l_knee_conf = get_pt_conf(13)
    r_knee, r_knee_conf = get_pt_conf(14)
    l_ankle, l_ankle_conf = get_pt_conf(15)
    r_ankle, r_ankle_conf = get_pt_conf(16)

    x1, y1, x2, y2 = bbox if bbox is not None else (0, 0, 1, 1)
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
    
    features = normalized_kpts + [angle, aspect_ratio]
    features_scaled = scaler.transform([features])
    
    pred = clf.predict(features_scaled)[0]
    pred_proba = clf.predict_proba(features_scaled)[0]
    
    status = 'Fallen' if pred == 1 else 'Standing'
    confidence = pred_proba[pred]

    # =========================================================================
    # GIẢI PHÁP 1: Lọc đối tượng đứng quá gần camera (Out-of-Frame Edge Filter)
    # =========================================================================
    is_touching_bottom = (y2 >= frame_height - 15)
    is_partially_clipped = is_touching_bottom and (l_hip_conf < 0.35 or r_hip_conf < 0.35 or l_ankle_conf < 0.35 or r_ankle_conf < 0.35)
    if is_partially_clipped or (is_touching_bottom and h > 0.75 * frame_height and aspect_ratio > 0.85):
        # Đứng quá gần camera nên bị cắt phần thân dưới -> Ép về Standing
        status = 'Standing'
        confidence = max(confidence, 0.85)

    # =========================================================================
    # GIẢI PHÁP 2: Kiểm tra vị trí Chân/Đầu gối so với Hông (Foot Grounding Check)
    # Phân biệt khom/cúi người nhặt đồ với nằm ngã thực sự dưới sàn
    # =========================================================================
    if status == 'Fallen':
        # Tính Y trung bình của Hông
        if l_hip_conf > 0.3 and r_hip_conf > 0.3:
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
        elif l_hip_conf > 0.3:
            hip_y = l_hip[1]
        elif r_hip_conf > 0.3:
            hip_y = r_hip[1]
        else:
            hip_y = None

        # Tính Y trung bình của Cổ chân hoặc Đầu gối
        if l_ankle_conf > 0.3 and r_ankle_conf > 0.3:
            feet_y = (l_ankle[1] + r_ankle[1]) / 2.0
        elif l_ankle_conf > 0.3:
            feet_y = l_ankle[1]
        elif r_ankle_conf > 0.3:
            feet_y = r_ankle[1]
        elif l_knee_conf > 0.3 or r_knee_conf > 0.3:
            feet_y = (l_knee[1] + r_knee[1]) / 2.0 if (l_knee_conf > 0.3 and r_knee_conf > 0.3) else (l_knee[1] if l_knee_conf > 0.3 else r_knee[1])
        else:
            feet_y = None

        # Nếu chân/đầu gối vẫn nằm ở phía dưới thấp hơn hông một khoảng rõ rệt (> 18% chiều cao box)
        # Trong OpenCV, Y tăng từ trên xuống dưới -> feet_y > hip_y nghĩa là chân đứng ở dưới hông
        if hip_y is not None and feet_y is not None:
            if (feet_y - hip_y) > 0.18 * h:
                # Đang khom người / cúi lưng nhặt đồ nhưng chân vẫn đứng trụ -> Ép về Standing
                status = 'Standing'
                confidence = max(confidence, 0.85)

    # Cứu cánh bằng Aspect Ratio cứng khi người nằm ngang thực sự và không bị chạm mép dưới
    if bbox is not None and aspect_ratio > 1.30 and not is_touching_bottom:
        status = 'Fallen'
        confidence = max(confidence, 0.85)
        
    return status, confidence

def main():
    # Sử dụng camera của laptop (mặc định src=0)
    print("Đang khởi động luồng đọc camera...")
    webcam = WebcamStream(src=0).start()
    
    output_width = 1000
    output_height = 600

    # Cấu hình bộ lọc thời gian chống báo giả (Nâng cấp 2)
    FALL_FRAME_THRESHOLD = 30
    track_db = {}
    next_track_id = 1
    MAX_LOST_FRAMES = 30

    print("\n==============================================")
    print("Hệ thống đã sẵn sàng!")
    print("Nhấn 'q' trực tiếp tại cửa sổ video để THOÁT.")
    print("==============================================\n")

    # Tính toán FPS hiển thị
    prev_time = 0

    while not webcam.stopped:
        frame = webcam.read()
        if frame is None:
            time.sleep(0.005) # Chờ một chút nếu hàng đợi trống
            continue

        frame = cv2.resize(frame, (output_width, output_height))

        # Nhận diện pose
        results = model.predict(frame, conf=0.25, iou=0.35, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        # 1. Tính toán Centroid
        current_centroids = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centroids.append((cx, cy))

        # 2. Khớp các Centroid
        assigned_ids = [None] * len(boxes)
        used_db_ids = set()

        for idx, (cx, cy) in enumerate(current_centroids):
            min_dist = float('inf')
            matched_id = None
            for db_id, info in track_db.items():
                if db_id in used_db_ids:
                    continue
                dcx, dcy = info['centroid']
                dist = np.sqrt((cx - dcx)**2 + (cy - dcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = db_id

            if matched_id is not None and min_dist < 180:
                assigned_ids[idx] = matched_id
                used_db_ids.add(matched_id)
                track_db[matched_id]['centroid'] = (cx, cy)
                track_db[matched_id]['lost_frames'] = 0
            else:
                new_id = next_track_id
                next_track_id += 1
                track_db[new_id] = {
                    'centroid': (cx, cy),
                    'fall_counter': 0,
                    'lost_frames': 0
                }
                assigned_ids[idx] = new_id
                used_db_ids.add(new_id)

        # 3. Cập nhật lost_frames
        for db_id in list(track_db.keys()):
            if db_id not in used_db_ids:
                track_db[db_id]['lost_frames'] += 1
                if track_db[db_id]['lost_frames'] > MAX_LOST_FRAMES:
                    del track_db[db_id]

        statuses = []
        confidences = []

        # Xử lý trạng thái và vẽ skeleton
        for i, keypoints in enumerate(keypoints_data):
            if len(keypoints) > 0 and i < len(boxes):
                box = boxes[i]
                status, confidence = check_fall_status_fallback(keypoints, bbox=box, frame_height=output_height)
                confidences.append(confidence)

                track_id = assigned_ids[i]
                if track_id is not None and track_id in track_db:
                    if status == 'Fallen':
                        track_db[track_id]['fall_counter'] += 1
                    else:
                        track_db[track_id]['fall_counter'] = 0

                    current_count = track_db[track_id]['fall_counter']
                    if current_count >= FALL_FRAME_THRESHOLD:
                        confirmed_status = 'Fallen (Confirmed)'
                    elif current_count > 0:
                        confirmed_status = f'Fallen (Pending {current_count}/{FALL_FRAME_THRESHOLD})'
                    else:
                        confirmed_status = 'Standing'
                else:
                    confirmed_status = status

                statuses.append(confirmed_status)
                draw_skeleton(frame, keypoints, confidence_threshold=0.5)
            else:
                statuses.append('Unknown')
                confidences.append(0.0)

        # Vẽ bounding box lên màn hình
        for i in range(min(len(boxes), len(statuses))):
            x1, y1, x2, y2 = boxes[i]
            status = statuses[i]
            conf_val = confidences[i]
            
            if 'Confirmed' in status:
                color = (0, 0, 255)
            elif 'Pending' in status:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            display_text = f"{status} {int(conf_val * 100)}%"
            cvzone.putTextRect(
                frame, display_text, (x1, y2 - 10),
                scale=1.5, thickness=2,
                colorT=(255, 255, 255), colorR=color,
                font=cv2.FONT_HERSHEY_PLAIN,
                offset=8,
                border=0
            )

        # Tính toán và vẽ FPS thực tế
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {int(fps_val)}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Hiển thị trực tiếp lên màn hình
        cv2.imshow('REAL-TIME Fall Detection (Multi-threaded)', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    webcam.stop()
    cv2.destroyAllWindows()
    print("Đã tắt camera và đóng ứng dụng.")

if __name__ == '__main__':
    main()
