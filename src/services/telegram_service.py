import os
import cv2
import time
import requests
import threading
import config
from src.services.db_service import DatabaseService

class TelegramService:
    """
    Dịch vụ gửi cảnh báo tự động qua Telegram Bot khi phát hiện người ngã (Fallen Confirmed).
    Tích hợp cơ chế:
    - Đa luồng (Background Thread) gửi ảnh không làm đứng/giật lag màn hình camera.
    - Thời gian chờ Cooldown chống spam tin nhắn liên tục.
    - Tự động ghi vào CSDL SQLite (bảng fall_events và subscribers).
    - Hỗ trợ gửi Broadcast tới nhiều người thân cùng lúc.
    """
    def __init__(self):
        self.enabled = config.TELEGRAM_ENABLED
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.cooldown = config.TELEGRAM_COOLDOWN_SECONDS
        self.last_sent_time = 0
        self.alert_dir = os.path.join(config.BASE_DIR, "video_result", "fall_alerts")
        os.makedirs(self.alert_dir, exist_ok=True)
        self.db = DatabaseService()

    def send_fall_alert(self, frame, status_text="Fallen (Confirmed)", confidence=1.0):
        # Kiểm tra công tắc kích hoạt và thông tin token
        if not self.enabled:
            return
        if not self.token or self.token == "YOUR_TELEGRAM_BOT_TOKEN":
            print("[Telegram Warning] TELEGRAM_ENABLED=True nhưng chưa nhập TELEGRAM_BOT_TOKEN trong config.py!")
            return

        current_time = time.time()
        # Cơ chế Cooldown chống spam
        if (current_time - self.last_sent_time) < self.cooldown:
            return

        self.last_sent_time = current_time

        # Khởi chạy luồng ngầm gửi ảnh để không làm giật lag luồng camera chính
        t = threading.Thread(target=self._send_async, args=(frame.copy(), status_text, confidence))
        t.daemon = True
        t.start()

    def _send_async(self, frame, status_text, confidence):
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"fall_alert_{timestamp_str}.jpg"
            img_path = os.path.join(self.alert_dir, filename)

            # 1. Lưu ảnh hiện trường ngã
            cv2.imwrite(img_path, frame)

            # 2. Ghi sự kiện vào CSDL SQLite
            event_id = self.db.log_fall_event(
                camera_source=config.CAMERA_SOURCE,
                status=status_text,
                confidence=confidence,
                image_path=img_path,
                alert_sent=1
            )

            # 3. Lấy danh sách tất cả các Chat ID đã đăng ký
            subscribers = self.db.get_active_subscribers()
            if not subscribers and self.chat_id:
                subscribers = [self.chat_id]

            caption = (
                f"🚨 CẢNH BÁO KHẨN CẤP: PHÁT HIỆN NGƯỜI NGÃ!\n\n"
                f"🆔 Mã sự kiện: #{event_id if event_id else 'N/A'}\n"
                f"⏰ Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📍 Vị trí/Camera: {config.CAMERA_SOURCE}\n"
                f"⚠️ Trạng thái: {status_text}\n"
                f"📸 Ảnh chụp hiện trường đính kèm bên dưới."
            )

            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            
            # Broadcast cảnh báo cho tất cả người nhận
            for target_chat_id in subscribers:
                try:
                    with open(img_path, 'rb') as photo_file:
                        payload = {'chat_id': target_chat_id, 'caption': caption}
                        files = {'photo': photo_file}
                        response = requests.post(url, data=payload, files=files, timeout=10)

                    if response.status_code == 200:
                        print(f"[Telegram Bot] Da gui anh canh bao hien truong thanh cong den Chat ID: {target_chat_id}")
                    else:
                        print(f"[Telegram Bot] Loi gui tin nhan den {target_chat_id} (Ma loi {response.status_code}): {response.text}")
                except Exception as req_err:
                    print(f"[Telegram Bot Error] Loi khi gui toi Chat ID {target_chat_id}: {req_err}")

        except Exception as e:
            print(f"\n[Telegram Bot Error]: {e}")

