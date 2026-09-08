import cv2
import os
import numpy as np
import cvzone
import config
from src.tracker import CentroidTracker
from src.detector import FallDetector, draw_skeleton

def main():
    video_path = 'video_test/test2.mp4'
    if not os.path.exists(video_path):
        video_path = 'test.mp4'

    if not os.path.exists(video_path):
        print(f"Error: Không tìm thấy file video mẫu '{video_path}'")
        return

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

    detector = FallDetector()
    tracker = CentroidTracker(
        max_distance=config.TRACKER_MAX_DISTANCE, 
        max_lost_frames=config.MAX_LOST_FRAMES
    )

    frame_count = 0
    show_window = True
    print("Bắt đầu xử lý video bằng mô hình Học Máy Modular. Nhấn 'q' để dừng...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (output_width, output_height))

        boxes, keypoints_data = detector.detect_poses(frame)
        assigned_ids = tracker.update(boxes)

        statuses = []
        confidences = []

        for i, keypoints in enumerate(keypoints_data):
            if len(keypoints) > 0 and i < len(boxes):
                box = boxes[i]
                track_id = assigned_ids[i]
                
                y_hip_norm_estimate = (box[1] + box[3]) / (2.0 * output_height)
                drop_velocity, had_sudden_drop = tracker.update_hip_position(track_id, y_hip_norm_estimate)

                status, confidence, actual_y_hip_norm = detector.predict_fall_status(
                    keypoints, 
                    bbox=box, 
                    frame_height=output_height,
                    had_sudden_drop=had_sudden_drop
                )
                
                if actual_y_hip_norm is not None:
                    tracker.update_hip_position(track_id, actual_y_hip_norm)

                confidences.append(confidence)

                is_fallen = (status == 'Fallen')
                fall_count = tracker.update_fall_counter(track_id, is_fallen)

                if fall_count >= config.FALL_FRAME_THRESHOLD:
                    confirmed_status = 'Fallen (Confirmed)'
                elif fall_count > 0:
                    confirmed_status = f'Fallen (Pending {fall_count}/{config.FALL_FRAME_THRESHOLD})'
                else:
                    confirmed_status = 'Standing'

                statuses.append(confirmed_status)
                draw_skeleton(frame, keypoints, confidence_threshold=0.5)
            else:
                statuses.append('Unknown')
                confidences.append(0.0)

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
                offset=10, border=0
            )

        out.write(frame)

        if show_window:
            try:
                cv2.imshow('YOLOv8 Pose & SVM Fall Detection (Modular)', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            except cv2.error:
                show_window = False

        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Xử lý hoàn thành! Kết quả lưu tại: '{output_path}'")

if __name__ == '__main__':
    main()
