import cv2
import numpy as np
import os
import sys

# Try importing ultralytics and cvzone, prompt installation if missing
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library is not installed. Please run: pip install ultralytics")
    sys.exit(1)

try:
    import cvzone
except ImportError:
    print("Error: 'cvzone' library is not installed. Please run: pip install cvzone")
    sys.exit(1)

def check_fall_status(keypoints):
    """
    Xác định trạng thái ngã và tính toán độ tin cậy của quyết định.
    Trả về: (status, confidence)
    - status: 'Fallen' hoặc 'Standing'
    - confidence: giá trị float từ 0.0 đến 1.0 (ví dụ 0.85 tương đương 85%)
    """
    # Các khớp xương chính được dùng trong quyết định:
    # 0: nose (đầu), 11: left_hip, 12: right_hip, 15: left_ankle, 16: right_ankle

    def get_pt_conf(idx):
        if idx >= len(keypoints):
            return np.array([0.0, 0.0]), 0.0
        pt = keypoints[idx][:2]
        conf = keypoints[idx][2] if len(keypoints[idx]) > 2 else 1.0
        return pt, conf

    nose, nose_conf = get_pt_conf(0)
    l_hip, l_hip_conf = get_pt_conf(11)
    r_hip, r_hip_conf = get_pt_conf(12)
    l_ankle, l_ankle_conf = get_pt_conf(15)
    r_ankle, r_ankle_conf = get_pt_conf(16)

    # 1. Tính toán độ tin cậy phát hiện khớp xương trung bình của mô hình YOLO
    keypoint_conf = (nose_conf + l_hip_conf + r_hip_conf + l_ankle_conf + r_ankle_conf) / 5.0

    # 2. Tính toán độ tin cậy hình học (Geometric Margin) dựa trên khoảng cách
    is_fall_detected = False
    max_fall_margin = 0.0

    # Điều kiện 1: Đầu thấp hơn hông (nose Y > hip Y)
    head_hip_diff_l = nose[1] - l_hip[1]
    head_hip_diff_r = nose[1] - r_hip[1]
    
    # Điều kiện 2: Hông thấp hơn cổ chân (hip Y > ankle Y)
    hip_ankle_diff_l = l_hip[1] - l_ankle[1]
    hip_ankle_diff_r = r_hip[1] - r_ankle[1]

    # Tìm biên độ lớn nhất chứng minh hành vi ngã
    if head_hip_diff_l > 0 and head_hip_diff_r > 0:
        is_fall_detected = True
        max_fall_margin = max(max_fall_margin, (head_hip_diff_l + head_hip_diff_r) / 2.0)
        
    if hip_ankle_diff_l > 0 or hip_ankle_diff_r > 0:
        is_fall_detected = True
        max_fall_margin = max(max_fall_margin, max(hip_ankle_diff_l, hip_ankle_diff_r))

    # Quy đổi biên độ hình học thành xác suất/độ tin cậy
    if is_fall_detected:
        # Nếu biên độ ngã lớn (ví dụ > 100 pixels), độ tin cậy hình học sẽ tiệm cận 1.0
        geom_conf = min(1.0, 0.5 + (max_fall_margin / 100.0) * 0.5)
        status = 'Fallen'
    else:
        # Nếu đứng, độ tin cậy đứng dựa trên việc đầu ở cao hơn hông bao nhiêu
        standing_margin_1 = min(l_hip[1] - nose[1], r_hip[1] - nose[1])
        standing_margin_2 = min(l_ankle[1] - l_hip[1], r_ankle[1] - r_hip[1])
        min_standing_margin = min(standing_margin_1, standing_margin_2)
        
        geom_conf = min(1.0, 0.5 + (max(0.0, min_standing_margin) / 100.0) * 0.5)
        status = 'Standing'

    # Kết hợp độ tin cậy nhận diện mô hình và độ tin cậy tư thế hình học
    final_conf = keypoint_conf * geom_conf
    final_conf = max(0.1, min(1.0, final_conf)) # Đảm bảo nằm trong khoảng [0.1, 1.0]
    
    return status, final_conf

def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    """
    Vẽ khung xương kết nối 17 điểm COCO với các màu khác nhau cho bên trái, bên phải và trung tâm.
    """
    color_left = (255, 255, 0)    # Cyan cho bên trái
    color_right = (0, 165, 255)   # Orange cho bên phải
    color_center = (0, 255, 0)    # Green cho đường trung tâm nối vai/hông

    # Các cặp khớp kết nối theo chuẩn COCO skeleton
    connections = [
        # Head (mắt, mũi, tai)
        (0, 1, color_left), (0, 2, color_right), 
        (1, 3, color_left), (2, 4, color_right),
        # Torso (vai, hông)
        (5, 6, color_center), (5, 11, color_center), 
        (6, 12, color_center), (11, 12, color_center),
        # Arms
        (5, 7, color_left), (7, 9, color_left), 
        (6, 8, color_right), (8, 10, color_right),
        # Legs
        (11, 13, color_left), (13, 15, color_left), 
        (12, 14, color_right), (14, 16, color_right)
    ]

    # Vẽ các liên kết xương
    for pt1_idx, pt2_idx, color in connections:
        if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
            continue
            
        pt1 = keypoints[pt1_idx]
        pt2 = keypoints[pt2_idx]
        
        conf1 = pt1[2] if len(pt1) > 2 else 1.0
        conf2 = pt2[2] if len(pt2) > 2 else 1.0
        
        if conf1 > confidence_threshold and conf2 > confidence_threshold:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            
    # Vẽ các nút khớp dạng chấm tròn đỏ
    for pt in keypoints:
        conf = pt[2] if len(pt) > 2 else 1.0
        if conf > confidence_threshold:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

def main():
    video_path = 'test.mp4'
    output_path = 'posees.mp4'
    model_name = 'yolov8n-pose.pt'  # Sử dụng model nano để tải và chạy nhanh

    if not os.path.exists(video_path):
        print(f"Error: Không tìm thấy file video mẫu '{video_path}' trong thư mục hiện tại.")
        return

    print(f"Đang tải mô hình YOLOv8-pose: {model_name}...")
    model = YOLO(model_name)

    print(f"Đang mở video: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    
    output_width = 1200
    output_height = 680

    print(f"Đang chuẩn bị ghi video kết quả vào '{output_path}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    frame_count = 0
    show_window = True

    print("Bắt đầu xử lý video. Nhấn 'q' để dừng sớm...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (output_width, output_height))

        # Nhận diện pose
        results = model.predict(frame, verbose=False)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data

        statuses = []
        confidences = []

        # Xử lý trạng thái và vẽ khung xương cho từng người
        for i, keypoints in enumerate(keypoints_data):
            if len(keypoints) > 0:
                # 1. Tính toán trạng thái đứng/ngã kèm độ tin cậy
                status, confidence = check_fall_status(keypoints)
                statuses.append(status)
                confidences.append(confidence)

                # 2. Vẽ khung xương COCO
                draw_skeleton(frame, keypoints, confidence_threshold=0.5)
            else:
                statuses.append('Unknown')
                confidences.append(0.0)

        # Vẽ bounding box và text hiển thị kết quả kèm độ tin cậy %
        for i in range(min(len(boxes), len(statuses))):
            x1, y1, x2, y2 = boxes[i]
            status = statuses[i]
            conf_val = confidences[i]
            
            # Chọn màu dựa trên trạng thái
            color = (0, 0, 255) if status == 'Fallen' else (0, 255, 0)
            
            # Vẽ hộp bao quanh
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ văn bản trạng thái: VD "Fallen 92%" hoặc "Standing 85%"
            display_text = f"{status} {int(conf_val * 100)}%"
            cvzone.putTextRect(
                frame, display_text, (x1, y2 - 10),
                scale=1.8, thickness=2,
                colorT=(255, 255, 255), colorR=color,
                font=cv2.FONT_HERSHEY_PLAIN,
                offset=10,
                border=0
            )

        # Ghi frame vào output video
        out.write(frame)

        # Hiển thị trực tiếp (nếu có GUI)
        if show_window:
            try:
                cv2.imshow('YOLOv8 Pose & Fall Detection', frame)
                if cv2.waitKey(1) == ord('q'):
                    print("Đã dừng xử lý video...")
                    break
            except cv2.error:
                show_window = False
                print("Headless environment detected. Tắt hiển thị trực quan và tiếp tục ghi file video...")

        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Xử lý hoàn thành! Kết quả đã lưu tại '{output_path}'")

if __name__ == '__main__':
    main()
