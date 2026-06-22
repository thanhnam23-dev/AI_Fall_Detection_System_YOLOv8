# Hướng dẫn Nâng cấp Hệ thống Phát hiện Ngã YOLOv8-pose

---

## 1. Nâng cấp 1: Tính toán Góc nghiêng cơ thể ($\theta$)

### Mục tiêu
Loại bỏ việc báo ngã sai khi một người đứng nhưng chỉ cúi gập lưng, khom lưng hoặc ngồi xổm.

### Nguyên lý hoạt động
Thay vì chỉ so sánh độ cao tương đối giữa đầu và hông, chúng ta sẽ dựng trục cơ thể bằng vector từ **Hông lên Vai** và đo góc lệch của vector này so với **Phương thẳng đứng (Trục Y)**.

```
            (Shoulder Center)
                 o
                / \
               /   \
              /  |  \
             /   |   \  <-- Góc θ so với phương thẳng đứng
            /    |    \
           /     |     \
          o-------------o
      Left Hip        Right Hip
            (Hip Center)
```

### Các bước triển khai trong Python
1.  **Tính Trung điểm Vai (Shoulder Center):**
    $$Shoulder_{mid} = \frac{Keypoint_5 + Keypoint_6}{2}$$
2.  **Tính Trung điểm Hông (Hip Center):**
    $$Hip_{mid} = \frac{Keypoint_{11} + Keypoint_{12}}{2}$$
3.  **Xác định Vector thân người:**
    $$\vec{u} = Shoulder_{mid} - Hip_{mid} = (x_{shoulder} - x_{hip}, y_{shoulder} - y_{hip})$$
4.  **Tính góc $\theta$ so với phương thẳng đứng $\vec{v} = (0, -1)$ hoặc $(0, 1)$:**
    Sử dụng hàm lượng giác trong thư viện `numpy`:
    ```python
    import numpy as np

    # Vector thân người u
    u = np.array([x_shoulder - x_hip, y_shoulder - y_hip])
    # Vector trục đứng v
    v = np.array([0, -1]) 
    
    # Tính cosine của góc
    cosine_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    ```
5.  **Ngưỡng kích hoạt:** Kích hoạt cảnh báo khi $\theta > 60^\circ$ (Trục thân người gần như nằm song song với mặt đất).

---

## 2. Nâng cấp 2: Bộ lọc thời gian chống báo giả (Temporal Validation)

### Mục tiêu
Tránh kích hoạt cảnh báo tức thời khi người dùng trượt chân nhẹ, khom người nhặt đồ rồi đứng dậy ngay lập tức.

### Nguyên lý hoạt động
Tạo một cơ chế trễ (delay filter). Một sự cố ngã chỉ được coi là hợp lệ nếu trạng thái ngã được mô hình phát hiện liên tục trong khoảng thời gian xác định trước.

```
[Frame 1: Fall] -> [Frame 2: Fall] -> ... -> [Frame 30: Fall] -> KÍCH HOẠT CẢNH BÁO
      |
  [Counter=1]            [Counter=2]                [Counter=30 (1 Giây)]
```

### Các bước triển khai trong Python
1.  Khai báo một biến đếm toàn cục `fall_counter = 0` ở đầu chương trình.
2.  Thiết lập ngưỡng khung hình, ví dụ ở video 30 FPS: `FALL_FRAME_THRESHOLD = 30` (tương đương duy trì trạng thái 1 giây).
3.  Trong vòng lặp xử lý frame:
    ```python
    if is_fall_detected:
        fall_counter += 1
    else:
        fall_counter = 0  # Reset ngay lập tức nếu đứng dậy
        
    if fall_counter >= FALL_FRAME_THRESHOLD:
        status = 'Fallen (Confirmed)'
        # Kích hoạt cảnh báo âm thanh hoặc gửi tin nhắn
    else:
        status = 'Standing'
    ```

---

## 3. Nâng cấp 3: Xử lý Đa luồng (Multi-threading) cơ bản

### Mục tiêu
Tách biệt luồng đọc camera và luồng chạy mô hình AI. Tránh hiện tượng video bị giật lag tích lũy do độ trễ xử lý (inference time) của mô hình YOLOv8.

### Nguyên lý hoạt động
*   **Thread 1 (Luồng đọc):** Chạy liên tục chỉ làm nhiệm vụ lấy frame mới nhất từ Camera/Video và đẩy vào hàng đợi `Queue`.
*   **Thread 2 (Luồng xử lý):** Lấy frame từ đầu hàng đợi `Queue` ra, chạy dự đoán YOLO và hiển thị. Nếu hàng đợi đầy, tự động giải phóng frame cũ để luôn có frame thời gian thực mới nhất.

```
+-------------------+      Frame      +------------------+      Frame      +------------------+
|   Video Capture   | --------------> |  Queue (max=30)  | --------------> |   YOLO & Logic   |
|    (Thread 1)     |                 |  (Thread-Safe)   |                 |    (Thread 2)    |
+-------------------+                 +------------------+                 +------------------+
```

### Các bước triển khai trong Python
Sử dụng thư viện `threading` và `queue` có sẵn:
```python
import threading
import queue
import time

frame_queue = queue.Queue(maxsize=30)

def video_read_thread(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01) # Chờ một chút nếu hàng đợi đầy

# Khởi chạy luồng đọc phụ
thread = threading.Thread(target=video_read_thread, args=(cap,))
thread.daemon = True
thread.start()

# Trong luồng chính, liên tục lấy frame xử lý:
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        # Chạy dự đoán YOLOv8-pose và hiển thị ở đây...
```

---

## 4. Nâng cấp 4: Tích hợp gửi cảnh báo qua Telegram Bot

### Mục tiêu
Gửi cảnh báo khẩn cấp có hình ảnh hiện trường trực tiếp tới điện thoại của người giám sát ngay khi sự cố ngã xảy ra.

### Các bước triển khai
1.  **Tạo Bot Telegram:**
    *   Mở Telegram, tìm kiếm `@BotFather`.
    *   Gửi lệnh `/newbot`, đặt tên cho Bot và nhận chuỗi `Token API` (Ví dụ: `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).
2.  **Lấy Chat ID:**
    *   Tạo một nhóm (Group) trên Telegram, thêm Bot vừa tạo vào nhóm.
    *   Mở trình duyệt truy cập: `https://api.telegram.org/bot<Token_Của_Bạn>/getUpdates` để tìm chuỗi `"id"` của nhóm chat (thường bắt đầu bằng dấu trừ, ví dụ: `-987654321`).
3.  **Lập trình gửi tin nhắn ảnh trong Python:**
    Sử dụng thư viện `requests` để gửi ảnh chụp màn hình (snapshot) khi phát hiện ngã:
    ```python
    import requests

    def send_telegram_alert(image_path):
        token = "TOKEN_BOT_CỦA_BẠN"
        chat_id = "CHAT_ID_NHÓM_CỦA_BẠN"
        
        # Nội dung tin nhắn định dạng Markdown
        message = "🚨 *CẢNH BÁO:* Phát hiện tai nạn ngã tại khu vực giám sát!"
        
        url_text = f"https://api.telegram.org/bot{token}/sendMessage"
        url_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
        
        # Gửi ảnh kèm chú thích
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': chat_id, 'caption': message, 'parse_mode': 'Markdown'}
            response = requests.post(url_photo, files=files, data=data)
            
        if response.status_code == 200:
            print("Đã gửi cảnh báo Telegram thành công!")
        else:
            print("Gửi cảnh báo thất bại:", response.text)
    ```
4.  **Tích hợp vào luồng xử lý:** Khi bộ lọc thời gian ở **Nâng cấp 2** xác nhận ngã thành công, gọi lệnh `cv2.imwrite('alert.jpg', frame)` để lưu ảnh tạm, sau đó gọi hàm `send_telegram_alert('alert.jpg')` để báo động.
