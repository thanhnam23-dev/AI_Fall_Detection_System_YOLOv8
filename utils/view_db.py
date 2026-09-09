import sqlite3
import os
import sys

# Đảm bảo đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "fall_events.db")

def view_database():
    if not os.path.exists(DB_PATH):
        print(f"[Loi] Khong tim thay file CSDL tai: {DB_PATH}")
        return

    print("=" * 70)
    print(f"   BO XEM DU LIEU CSDL SQLITE (FALL_EVENTS.DB)")
    print("=" * 70)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. Xem danh sach nguoi dang ky nhan canh bao (subscribers)
        print("\n[1] DANH SACH NGUOI NHAN CANH BAO (Biol: subscribers):")
        print("-" * 70)
        cursor.execute("SELECT id, name, chat_id, email, role, is_active FROM subscribers")
        subscribers = cursor.fetchall()
        if subscribers:
            print(f"{'ID':<4} | {'Ten Nguoi Nhan':<22} | {'Telegram Chat ID':<18} | {'Email':<20} | {'Trang Thai'}")
            print("-" * 70)
            for sub in subscribers:
                status = "Kich hoat" if sub[5] == 1 else "Tam dung"
                print(f"{sub[0]:<4} | {str(sub[1]):<22} | {str(sub[2]):<18} | {str(sub[3]):<20} | {status}")
        else:
            print("Chua co nguoi dung nao dang ky.")

        # 2. Xem lich su su kien nga (fall_events)
        print("\n[2] LICH SU SU KIEN NGA (Biol: fall_events):")
        print("-" * 70)
        cursor.execute("SELECT id, timestamp, camera_source, status, confidence, telegram_sent, email_sent FROM fall_events ORDER BY id DESC LIMIT 20")
        events = cursor.fetchall()
        if events:
            print(f"{'ID':<4} | {'Thoi Gian':<19} | {'Camera':<8} | {'Trang Thai':<18} | {'Telegram':<8} | {'Email'}")
            print("-" * 70)
            for ev in events:
                tg_sent = "Thanh cong" if ev[5] == 1 else "That bai"
                em_sent = "Thanh cong" if ev[6] == 1 else "That bai"
                print(f"{ev[0]:<4} | {ev[1]:<19} | {str(ev[2]):<8} | {ev[3]:<18} | {tg_sent:<8} | {em_sent}")
        else:
            print("Chua co su kien nga nao duoc ghi nhan trong CSDL.")

        print("=" * 70)
        conn.close()

    except Exception as e:
        print(f"[Loi doc CSDL]: {e}")

if __name__ == "__main__":
    view_database()
