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

def is_fallen(keypoints):
    """
    Xác định xem người đó có bị ngã hay không dựa trên các điểm keypoints.
    Logic cơ bản trong notebook:
    - So sánh tọa độ Y (trong OpenCV, Y tăng dần từ trên xuống dưới).
    - Nếu Đầu (Nose - keypoint 0) nằm thấp hơn Hông (Hips - keypoint 11, 12).
    - Hoặc Hông nằm thấp hơn Cổ chân (Ankles - keypoint 15, 16).
    """
    # Lấy tọa độ (x, y) của các điểm cần thiết
    head = keypoints[0][:2]  # Nose (Mũi)
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    left_ankle = keypoints[15][:2]
    right_ankle = keypoints[16][:2]

    # Kiểm tra xem đầu có gần mặt đất hơn hông không (Y lớn hơn tức là nằm dưới)
    if head[1] > left_hip[1] and head[1] > right_hip[1]:
        return True

    # Kiểm tra xem hông có gần mặt đất hơn cổ chân không
    if left_hip[1] > left_ankle[1] or right_hip[1] > right_ankle[1]:
        return True

    return False

def main():
    video_path = 'test.mp4'
    output_path = 'posees.mp4'
    model_name = 'yolov8n-pose.pt'  # Sử dụng model nano cho nhẹ và chạy nhanh, bạn có thể đổi sang yolov8x-pose.pt

    if not os.path.exists(video_path):
        print(f"Error: Không tìm thấy file video mẫu '{video_path}' trong thư mục hiện tại.")
        return

    print(f"Đang tải mô hình YOLOv8-pose: {model_name}...")
    model = YOLO(model_name)

    print(f"Đang mở video: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    # Đọc thông số video để thiết lập VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Thiết lập kích thước output giống notebook (1200x680) hoặc giữ nguyên
    output_width = 1200
    output_height = 680

    print(f"Đang chuẩn bị ghi video kết quả vào '{output_path}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Sử dụng codec mp4v thông dụng
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    frame_count = 0
    show_window = True

    print("Bắt đầu xử lý video. Nhấn 'q' để dừng sớm...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Resize frame
        frame = cv2.resize(frame, (output_width, output_height))

        # Thực hiện dự đoán pose
        # verbose=False để giảm log rác trên terminal
        results = model.predict(frame, verbose=False)

        # Trích xuất bounding boxes và keypoints
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.data

        statuses = []

        # Kiểm tra trạng thái ngã của từng người được phát hiện
        for i, keypoints in enumerate(keypoints_data):
            if len(keypoints) > 0:
                if is_fallen(keypoints):
                    statuses.append('Fallen')
                else:
                    statuses.append('Standing')
            else:
                statuses.append('Unknown')

        # Vẽ bounding box và trạng thái lên frame
        for i in range(min(len(boxes), len(statuses))):
            x1, y1, x2, y2 = boxes[i]
            
            # Nếu ngã vẽ màu đỏ (0, 0, 255), đứng vẽ màu xanh lá (0, 255, 0)
            color = (0, 0, 255) if statuses[i] == 'Fallen' else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dùng cvzone vẽ text đẹp hơn
            cvzone.putTextRect(
                frame, f"{statuses[i]}", (x1, y2 - 10),
                scale=2, thickness=2,
                colorT=(255, 255, 255), colorR=color,
                font=cv2.FONT_HERSHEY_PLAIN,
                offset=10,
                border=0
            )

        # Ghi frame vào output video
        out.write(frame)

        # Hiển thị frame lên màn hình (nếu có môi trường GUI)
        if show_window:
            try:
                cv2.imshow('YOLOv8 Fall Detection Template', frame)
                if cv2.waitKey(1) == ord('q'):
                    print("Đã nhấn 'q'. Đang dừng...")
                    break
            except cv2.error:
                # Nếu chạy ở môi trường headless (không có màn hình GUI)
                show_window = False
                print("Headless environment detected. Tắt hiển thị trực quan và tiếp tục ghi file video...")

        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Hoàn thành! Kết quả đã được lưu tại '{output_path}'")

if __name__ == '__main__':
    main()
