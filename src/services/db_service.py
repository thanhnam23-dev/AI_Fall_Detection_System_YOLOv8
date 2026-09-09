import os
import sqlite3
import time
import config

class DatabaseService:
    """
    Quản lý lưu trữ sự kiện ngã và danh sách người nhận cảnh báo đa nền tảng (Telegram & Email).
    Sử dụng SQLite làm CSDL embedded mỏng nhẹ, không cần cài đặt server.
    """
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        self.enabled = getattr(config, 'DB_ENABLED', True)
        if self.enabled:
            self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Khởi tạo bảng fall_events và subscribers nếu chưa tồn tại."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. Bảng lịch sử sự kiện ngã (hỗ trợ lưu vết trạng thái Telegram & Email)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fall_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        camera_source TEXT NOT NULL,
                        status TEXT NOT NULL,
                        confidence REAL,
                        image_path TEXT,
                        telegram_sent INTEGER DEFAULT 1,
                        email_sent INTEGER DEFAULT 0
                    )
                """)

                # 2. Bảng quản lý người đăng ký nhận cảnh báo đa nền tảng (Telegram & Email)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS subscribers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        chat_id TEXT UNIQUE,
                        email TEXT UNIQUE,
                        phone TEXT,
                        role TEXT DEFAULT 'Family',
                        registered_at TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1
                    )
                """)
                
                # Thêm mặc định chat_id từ config nếu chưa có
                if hasattr(config, 'TELEGRAM_CHAT_ID') and config.TELEGRAM_CHAT_ID:
                    cursor.execute("""
                        INSERT OR IGNORE INTO subscribers (name, chat_id, email, role, registered_at, is_active)
                        VALUES (?, ?, ?, ?, ?, 1)
                    """, ("Chu ho / Quan tri vien", str(config.TELEGRAM_CHAT_ID), getattr(config, 'DEFAULT_ALERT_EMAIL', 'admin@example.com'), "Admin", time.strftime("%Y-%m-%d %H:%M:%S")))

                conn.commit()
                print(f"[DB Service] Khoi tao CSDL SQLite thanh cong tai: {self.db_path}")
        except Exception as e:
            print(f"[DB Service Error] Khoi tao CSDL that bai: {e}")

    def log_fall_event(self, camera_source, status="FALLEN CONFIRMED", confidence=1.0, image_path="", telegram_sent=1, email_sent=0):
        """Ghi nhận sự kiện ngã vào CSDL."""
        if not self.enabled:
            return None
        try:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO fall_events (timestamp, camera_source, status, confidence, image_path, telegram_sent, email_sent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp_str, str(camera_source), status, float(confidence), image_path, int(telegram_sent), int(email_sent)))
                conn.commit()
                event_id = cursor.lastrowid
                print(f"[DB Service] Da ghi nhan su kien nga ID #{event_id} vao CSDL.")
                return event_id
        except Exception as e:
            print(f"[DB Service Error] Loi ghi CSDL: {e}")
            return None

    def add_subscriber(self, name, chat_id="", email="", phone="", role="Family"):
        """Thêm người nhận cảnh báo qua Telegram và/hoặc Email."""
        if not self.enabled:
            return False
        try:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO subscribers (name, chat_id, email, phone, role, registered_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (name, str(chat_id) if chat_id else None, str(email) if email else None, phone, role, timestamp_str))
                conn.commit()
                print(f"[DB Service] Da dang ky nguoi dung {name} (Chat ID: {chat_id}, Email: {email}) thanh cong.")
                return True
        except Exception as e:
            print(f"[DB Service Error] Loi them nguoi dung: {e}")
            return False

    def get_active_subscribers(self):
        """Lấy danh sách các Telegram Chat ID đang kích hoạt để broadcast cảnh báo."""
        if not self.enabled:
            return [config.TELEGRAM_CHAT_ID] if hasattr(config, 'TELEGRAM_CHAT_ID') else []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chat_id FROM subscribers WHERE is_active = 1 AND chat_id IS NOT NULL AND chat_id != ''")
                rows = cursor.fetchall()
                chat_ids = [r[0] for r in rows if r[0]]
                if not chat_ids and hasattr(config, 'TELEGRAM_CHAT_ID') and config.TELEGRAM_CHAT_ID:
                    chat_ids = [config.TELEGRAM_CHAT_ID]
                return chat_ids
        except Exception as e:
            print(f"[DB Service Error] Loi lay danh sach subscribers Telegram: {e}")
            return [config.TELEGRAM_CHAT_ID] if hasattr(config, 'TELEGRAM_CHAT_ID') else []

    def get_active_emails(self):
        """Lấy danh sách địa chỉ Email đang kích hoạt để gửi báo cáo cảnh báo."""
        if not self.enabled:
            return []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT email FROM subscribers WHERE is_active = 1 AND email IS NOT NULL AND email != ''")
                rows = cursor.fetchall()
                emails = [r[0] for r in rows if r[0]]
                return emails
        except Exception as e:
            print(f"[DB Service Error] Loi lay danh sach email subscribers: {e}")
            return []

    def get_recent_events(self, limit=20):
        """Lấy danh sách sự kiện ngã gần đây."""
        if not self.enabled:
            return []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, timestamp, camera_source, status, confidence, image_path, telegram_sent, email_sent FROM fall_events ORDER BY id DESC LIMIT ?", (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"[DB Service Error] Loi truy van su kien: {e}")
            return []
