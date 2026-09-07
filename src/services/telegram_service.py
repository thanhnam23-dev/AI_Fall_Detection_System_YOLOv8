import os
import cv2
import time
import requests
import threading
import config

class TelegramService:
    """
    Dịch vụ gửi cảnh báo tự động qua Telegram Bot khi phát hiện người ngã (Fallen Confirmed).
    Tích hợp cơ chế:
    - Đa luồng (Background Thread) gửi ảnh không làm đứng/giật lag màn hình camera.
    - Thời gian chờ Cooldown chống spam tin nhắn liên tục.
    - Lưu ảnh chụp hiện trường vào thư mục video_result/fall_alerts/.
    """
    def __init__(self):
        self.enabled = config.TELEGRAM_ENABLED
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.cooldown = config.TELEGRAM_COOLDOWN_SECONDS
        self.last_sent_time = 0
        self.alert_dir = os.path.join(config.BASE_DIR, "video_result", "fall_alerts")
        os.makedirs(self.alert_dir, exist_ok=True)

    def send_fall_alert(self, frame, status_text="Fallen (Confirmed)"):
        # Kiểm tra công tắc kích hoạt và thông tin token
        if not self.enabled:
            return
        if not self.token or not self.chat_id or self.token == "YOUR_TELEGRAM_BOT_TOKEN":
            print("[Telegram Warning] TELEGRAM_ENABLED=True nhưng chưa nhập TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong config.py!")
            return

        current_time = time.time()
        # Cơ chế Cooldown chống spam
        if (current_time - self.last_sent_time) < self.cooldown:
            return

        self.last_sent_time = current_time

        # Khởi chạy luồng ngầm gửi ảnh để không làm giật lag luồng camera chính
        t = threading.Thread(target=self._send_async, args=(frame.copy(), status_text))
        t.daemon = True
        t.start()

    def _send_async(self, frame, status_text):
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"fall_alert_{timestamp_str}.jpg"
            img_path = os.path.join(self.alert_dir, filename)

            # Lưu ảnh hiện trường ngã
            cv2.imwrite(img_path, frame)

            caption = (
                f"🚨 CẢNH BÁO KHẨN CẤP: PHÁT HIỆN NGƯỜI NGÃ!\n\n"
                f"⏰ Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📍 Vị trí/Camera: {config.CAMERA_SOURCE}\n"
                f"⚠️ Trạng thái: {status_text}\n"
                f"📸 Ảnh chụp hiện trường đính kèm bên dưới."
            )

            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            with open(img_path, 'rb') as photo_file:
                payload = {'chat_id': self.chat_id, 'caption': caption}
                files = {'photo': photo_file}
                response = requests.post(url, data=payload, files=files, timeout=10)

            if response.status_code == 200:
                print(f"\n[Telegram Bot] 🟢 Đã gửi ảnh cảnh báo hiện trường thành công đến Chat ID: {self.chat_id}")
            else:
                print(f"\n[Telegram Bot] 🔴 Lỗi gửi tin nhắn (Mã lỗi {response.status_code}): {response.text}")
        except Exception as e:
            print(f"\n[Telegram Bot Error]: {e}")
