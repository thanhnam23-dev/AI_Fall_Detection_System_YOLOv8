import cv2
import threading
import queue
import time

class WebcamStream:
    """
    Lớp Đa luồng đọc Camera (Nâng cấp 3).
    Tách biệt luồng đọc camera và luồng xử lý AI.
    Duy trì hàng đợi tối đa 3 frames để giải phóng bộ đệm, luôn giữ frame mới nhất (Zero Accumulated Delay).
    Hỗ trợ cả Camera Laptop (src=0) và Camera IP Imou qua RTSP.
    """
    def __init__(self, src=0, queue_size=3):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break

            # Nếu hàng đợi đầy, đẩy bớt frame cũ ra để nhường chỗ cho frame mới nhất
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if self.q.empty():
            return None
        return self.q.get()

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
