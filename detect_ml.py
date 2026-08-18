import cv2
import numpy as np
import os
import sys
import pickle
import cvzone
from ultralytics import YOLO

# Khởi chạy mô hình YOLOv8-pose
YOLO_PATH = "models/yolov8n-pose.pt" if os.path.exists("models/yolov8n-pose.pt") else "yolov8n-pose.pt"
try:
    model = YOLO(YOLO_PATH)
except Exception as e:
    print(f"Error loading YOLO: {e}")
    sys.exit(1)

# Đường dẫn file mô hình ML và Scaler đã huấn luyện
MODEL_PATH = "models/fall_classifier.pkl" if os.path.exists("models/fall_classifier.pkl") else "fall_classifier.pkl"
SCALER_PATH = "models/scaler.pkl" if os.path.exists("models/scaler.pkl") else "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Error: Không tìm thấy file mô hình SVM trong 'models/'. Vui lòng huấn luyện mô hình trước!")
    sys.exit(1)

# Tải mô hình và bộ chuẩn hóa
with open(MODEL_PATH, 'rb') as f:
    clf = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Hàm tính góc nghiêng cơ thể phục vụ trích xuất đặc trưng
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
    color_left = (255, 255, 0)    # Cyan
    color_right = (0, 165, 255)   # Orange
    color_center = (0, 255, 0)    # Green
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
def check_fall_status_fallback(keypoints, bbox=None, frame_height=680):
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

    # GIẢI PHÁP 1: Lọc đối tượng đứng quá gần camera (Out-of-Frame Edge Filter)
    is_touching_bottom = (y2 >= frame_height - 15)
    is_partially_clipped = is_touching_bottom and (l_hip_conf < 0.35 or r_hip_conf < 0.35 or l_ankle_conf < 0.35 or r_ankle_conf < 0.35)
    if is_partially_clipped or (is_touching_bottom and h > 0.75 * frame_height and aspect_ratio > 0.85):
        status = 'Standing'
        confidence = max(confidence, 0.85)

    # GIẢI PHÁP 2: Kiểm tra vị trí Chân/Đầu gối so với Hông (Foot Grounding Check)
    if status == 'Fallen':
        if l_hip_conf > 0.3 and r_hip_conf > 0.3:
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
        elif l_hip_conf > 0.3:
            hip_y = l_hip[1]
        elif r_hip_conf > 0.3:
            hip_y = r_hip[1]
        else:
            hip_y = None

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

        if hip_y is not None and feet_y is not None:
            if (feet_y - hip_y) > 0.18 * h:
                status = 'Standing'
                confidence = max(confidence, 0.85)

    if bbox is not None and aspect_ratio > 1.30 and not is_touching_bottom:
        status = 'Fallen'
        confidence = max(confidence, 0.85)
        
    return status, confidence

def main():
    # Sử dụng video test2 để kiểm chứng
    video_path = 'video_test/test2.mp4'
    if not os.path.exists(video_path):
        video_path = 'test.mp4'

    if not os.path.exists(video_path):
        print(f"Error: Không tìm thấy file video mẫu '{video_path}'")
        return

    # Đường dẫn file đầu ra
    output_path = 'video_result/posees2_ml.mp4'
    
    if not os.path.exists('video_result'):
        os.makedirs('video_result')

    print(f"Đang mở video: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    
    output_width = 1200
    output_height = 680

    print(f"Đang ghi video kết quả vào '{output_path}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Cấu hình bộ lọc thời gian chống báo giả (Nâng cấp 2)
    FALL_FRAME_THRESHOLD = 30
    
    # Centroid Tracker để ổn định ID
    track_db = {}
    next_track_id = 1
    MAX_LOST_FRAMES = 30

    frame_count = 0
    show_window = True

    print("Bắt đầu xử lý video bằng mô hình Học Máy. Nhấn 'q' để dừng sớm...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (output_width, output_height))

        # Nhận diện pose bằng YOLO
        results = model.predict(frame, conf=0.25, iou=0.35, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        # 1. Tính toán Centroid cho frame hiện tại
        current_centroids = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centroids.append((cx, cy))

        # 2. Khớp các Centroid với track_db
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

        # Xử lý trạng thái dựa trên mô hình SVM
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

        # Vẽ bounding box
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
                scale=1.8, thickness=2,
                colorT=(255, 255, 255), colorR=color,
                font=cv2.FONT_HERSHEY_PLAIN,
                offset=10,
                border=0
            )

        out.write(frame)

        if show_window:
            try:
                cv2.imshow('YOLOv8 Pose & SVM Fall Detection', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            except cv2.error:
                show_window = False
                print("Headless environment detected. Ghi file video...")

        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Xử lý hoàn thành! Kết quả lưu tại: '{output_path}'")

if __name__ == '__main__':
    main()
