import cv2
import time
import cvzone
import config
from src.stream import WebcamStream
from src.tracker import CentroidTracker
from src.detector import FallDetector, draw_skeleton
from src.services.telegram_service import TelegramService

def main():
    print("=====================================================")
    print(" HỆ THỐNG PHÁT HIỆN NGÃ AI (REAL-TIME LIVE STREAM)")
    print(f" Nguồn Camera: {config.CAMERA_SOURCE}")
    print("=====================================================\n")

    # 1. Khởi tạo các Module
    print("Đang khởi động luồng đọc camera đa luồng...")
    webcam = WebcamStream(src=config.CAMERA_SOURCE).start()
    
    print("Đang khởi tạo mô hình AI, Tracker & Telegram Service...")
    detector = FallDetector()
    tracker = CentroidTracker(
        max_distance=config.TRACKER_MAX_DISTANCE, 
        max_lost_frames=config.MAX_LOST_FRAMES
    )
    telegram = TelegramService()

    print("\nHệ thống đã sẵn sàng! Nhấn 'q' tại cửa sổ video để THOÁT.\n")
    prev_time = time.time()

    while not webcam.stopped:
        frame = webcam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.resize(frame, (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT))

        # 2. Phát hiện Poses từ YOLOv8-pose
        boxes, keypoints_data = detector.detect_poses(frame)

        # 3. Cập nhật Tracker giữ ổn định ID
        assigned_ids = tracker.update(boxes)

        statuses = []
        confidences = []

        # 4. Phân loại bằng SVM + Bộ lọc chống báo nhầm
        for i, keypoints in enumerate(keypoints_data):
            if len(keypoints) > 0 and i < len(boxes):
                box = boxes[i]
                status, confidence = detector.predict_fall_status(
                    keypoints, 
                    bbox=box, 
                    frame_height=config.OUTPUT_HEIGHT
                )
                confidences.append(confidence)

                # Cập nhật đếm trễ 30 frames
                track_id = assigned_ids[i]
                is_fallen = (status == 'Fallen')
                fall_count = tracker.update_fall_counter(track_id, is_fallen)

                if fall_count >= config.FALL_FRAME_THRESHOLD:
                    confirmed_status = 'Fallen (Confirmed)'
                    # Kích hoạt gửi ảnh cảnh báo Telegram
                    telegram.send_fall_alert(frame, confirmed_status)
                elif fall_count > 0:
                    confirmed_status = f'Fallen (Pending {fall_count}/{config.FALL_FRAME_THRESHOLD})'
                else:
                    confirmed_status = 'Standing'

                statuses.append(confirmed_status)
                draw_skeleton(frame, keypoints, confidence_threshold=0.5)
            else:
                statuses.append('Unknown')
                confidences.append(0.0)

        # 5. Vẽ Bounding box & nhãn lên màn hình
        for i in range(min(len(boxes), len(statuses))):
            x1, y1, x2, y2 = boxes[i]
            status = statuses[i]
            conf_val = confidences[i]
            
            if 'Confirmed' in status:
                color = (0, 0, 255)       # Đỏ
            elif 'Pending' in status:
                color = (0, 165, 255)     # Cam
            else:
                color = (0, 255, 0)       # Xanh lá
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            display_text = f"{status} {int(conf_val * 100)}%"
            cvzone.putTextRect(
                frame, display_text, (x1, y2 - 10),
                scale=1.5, thickness=2,
                colorT=(255, 255, 255), colorR=color,
                font=cv2.FONT_HERSHEY_PLAIN,
                offset=8, border=0
            )

        # Tính toán & Vẽ FPS thực tế
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {int(fps_val)}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow('REAL-TIME Fall Detection System (Modular)', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    webcam.stop()
    cv2.destroyAllWindows()
    print("Đã đóng hệ thống thành công.")

if __name__ == '__main__':
    main()
