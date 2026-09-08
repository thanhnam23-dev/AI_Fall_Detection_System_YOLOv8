import cv2
import numpy as np
import os
import sys
import pickle
from ultralytics import YOLO
import config

def calculate_angle(keypoints):
    """Tính góc nghiêng trục cơ thể so với phương thẳng đứng"""
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
    """Vẽ bộ khung xương 17 khớp của YOLOv8-pose"""
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

class FallDetector:
    """
    Bộ nhận diện ngã AI kết hợp YOLOv8-pose + SVM Classifier.
    Tích hợp các thuật toán nâng cao:
    - Giải pháp 1: Lọc đối tượng đứng sát camera (Out-of-Frame Edge Filter)
    - Giải pháp 2: Lọc tư thế cúi nhặt đồ (Foot Grounding Check)
    - CẢI TIẾN MỚI 1: Nhận diện ngã khép chân/co quắp (Hip Screen Position + Tilt Angle)
    - CẢI TIẾN MỚI 2: Phân biệt ngã/đột quỵ đột ngột với nằm từ từ lên giường (Sudden Drop Velocity)
    """
    def __init__(self):
        # 1. Ưu tiên tải YOLOv8-pose Small (chính xác cao hơn cho nhiều người), fallback về Nano
        yolo_path = config.YOLO_MODEL_PATH
        print(f"Đang khởi tạo mô hình YOLOv8-pose từ: '{yolo_path}'...")
        try:
            self.yolo_model = YOLO(yolo_path)
        except Exception as e:
            print(f"Lỗi tải mô hình YOLO ({yolo_path}): {e}")
            sys.exit(1)

        # 2. Tải SVM Classifier & Scaler
        if not os.path.exists(config.SVM_MODEL_PATH) or not os.path.exists(config.SCALER_MODEL_PATH):
            print(f"Lỗi: Không tìm thấy file mô hình SVM ({config.SVM_MODEL_PATH}). Vui lòng huấn luyện mô hình trước!")
            sys.exit(1)

        with open(config.SVM_MODEL_PATH, 'rb') as f:
            self.clf = pickle.load(f)
        with open(config.SCALER_MODEL_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

    def detect_poses(self, frame):
        """Phát hiện Bounding Boxes và Keypoints bằng YOLOv8-pose"""
        results = self.yolo_model.predict(
            frame, 
            conf=config.YOLO_CONF_THRESHOLD, 
            iou=config.YOLO_IOU_THRESHOLD, 
            verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        return boxes, keypoints_data

    def predict_fall_status(self, keypoints, bbox, frame_height=600, had_sudden_drop=False):
        """Dự đoán trạng thái tư thế và áp dụng các bộ lọc chống báo nhầm nâng cao"""
        def get_pt_conf(idx):
            if idx >= len(keypoints):
                return np.array([0.0, 0.0]), 0.0
            pt = keypoints[idx][:2]
            conf = keypoints[idx][2] if len(keypoints[idx]) > 2 else 1.0
            return pt, conf

        l_hip, l_hip_conf = get_pt_conf(11)
        r_hip, r_hip_conf = get_pt_conf(12)
        l_knee, l_knee_conf = get_pt_conf(13)
        r_knee, r_knee_conf = get_pt_conf(14)
        l_ankle, l_ankle_conf = get_pt_conf(15)
        r_ankle, r_ankle_conf = get_pt_conf(16)

        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Lấy tọa độ Y hông chuẩn hóa theo màn hình
        if l_hip_conf > 0.3 and r_hip_conf > 0.3:
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
        elif l_hip_conf > 0.3:
            hip_y = l_hip[1]
        elif r_hip_conf > 0.3:
            hip_y = r_hip[1]
        else:
            hip_y = (y1 + y2) / 2.0
            
        y_hip_norm = hip_y / max(1.0, float(frame_height))
        
        # Chuẩn hóa tọa độ 17 khớp xương về khoảng [0, 1] relative với bounding box
        normalized_kpts = []
        for kp in keypoints:
            kx, ky = kp[0], kp[1]
            kx_norm = (kx - x1) / w
            ky_norm = (ky - y1) / h
            normalized_kpts.extend([kx_norm, ky_norm])
        
        angle = calculate_angle(keypoints)
        aspect_ratio = w / h
        
        features = normalized_kpts + [angle, aspect_ratio]
        features_scaled = self.scaler.transform([features])
        
        pred = self.clf.predict(features_scaled)[0]
        pred_proba = self.clf.predict_proba(features_scaled)[0]
        
        status = 'Fallen' if pred == 1 else 'Standing'
        confidence = pred_proba[pred]

        # GIẢI PHÁP 1: Lọc đối tượng đứng quá gần camera (Out-of-Frame Edge Filter)
        is_touching_bottom = (y2 >= frame_height - 15)
        is_partially_clipped = is_touching_bottom and (l_hip_conf < 0.35 or r_hip_conf < 0.35 or l_ankle_conf < 0.35 or r_ankle_conf < 0.35)
        if is_partially_clipped or (is_touching_bottom and h > 0.75 * frame_height and aspect_ratio > 0.85):
            status = 'Standing'
            confidence = max(confidence, 0.85)

        # CẢI TIẾN MỚI 1 (FIX BUG NGÃ KHÉP CHÂN):
        # Nếu góc nghiêng cơ thể > 45 độ VÀ hông nằm ở nửa dưới màn hình (y_hip_norm > 0.50)
        # -> Kích hoạt Fallen kể cả khi 2 chân khép sát (Aspect Ratio < 1.0)
        is_legs_together_fall = (angle > 45.0) and (y_hip_norm > config.HIP_LOWER_SCREEN_RATIO) and not is_touching_bottom
        if is_legs_together_fall:
            status = 'Fallen'
            confidence = max(confidence, 0.88)

        # GIẢI PHÁP 2: Kiểm tra vị trí Chân/Đầu gối so với Hông (Foot Grounding Check)
        if status == 'Fallen':
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
            if hip_y is not None and feet_y is not None:
                if (feet_y - hip_y) > 0.18 * h:
                    status = 'Standing'
                    confidence = max(confidence, 0.85)

        # Cứu cánh bằng Aspect Ratio cứng khi nằm ngã dang chân thực sự
        if aspect_ratio > 1.30 and not is_touching_bottom:
            status = 'Fallen'
            confidence = max(confidence, 0.85)

        # CẢI TIẾN MỚI 2 (FIX BUG NẰM NGỦ VS NGÃ/ĐỘT QUỤY):
        # Nếu đang ở vị trí nằm nghiêng/ngửa nhưng KHÔNG CÓ cú rơi tự do vận tốc nhanh (had_sudden_drop == False)
        # -> Đây là nằm từ từ lên giường/sofa ngủ -> Không tính là ngã!
        if status == 'Fallen' and not had_sudden_drop and not is_legs_together_fall:
            status = 'Standing'  # Hoặc tư thế nằm nghỉ sinh hoạt
            confidence = max(confidence, 0.80)

        return status, confidence, y_hip_norm
