import numpy as np

class CentroidTracker:
    """
    Bộ theo dõi tâm đối tượng hình học (Centroid Tracker).
    Giúp giữ ổn định ID người ngã giữa các frame liên tiếp, triệt tiêu hiện tượng nhảy ID (ID Switch).
    """
    def __init__(self, max_distance=180, max_lost_frames=30):
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.track_db = {}
        self.next_track_id = 1

    def update(self, boxes):
        current_centroids = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centroids.append((cx, cy))

        assigned_ids = [None] * len(boxes)
        used_db_ids = set()

        for idx, (cx, cy) in enumerate(current_centroids):
            min_dist = float('inf')
            matched_id = None
            for db_id, info in self.track_db.items():
                if db_id in used_db_ids:
                    continue
                dcx, dcy = info['centroid']
                dist = np.sqrt((cx - dcx)**2 + (cy - dcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = db_id

            if matched_id is not None and min_dist < self.max_distance:
                assigned_ids[idx] = matched_id
                used_db_ids.add(matched_id)
                self.track_db[matched_id]['centroid'] = (cx, cy)
                self.track_db[matched_id]['lost_frames'] = 0
            else:
                new_id = self.next_track_id
                self.next_track_id += 1
                self.track_db[new_id] = {
                    'centroid': (cx, cy),
                    'fall_counter': 0,
                    'lost_frames': 0
                }
                assigned_ids[idx] = new_id
                used_db_ids.add(new_id)

        for db_id in list(self.track_db.keys()):
            if db_id not in used_db_ids:
                self.track_db[db_id]['lost_frames'] += 1
                if self.track_db[db_id]['lost_frames'] > self.max_lost_frames:
                    del self.track_db[db_id]

        return assigned_ids

    def update_fall_counter(self, track_id, is_fallen):
        if track_id is not None and track_id in self.track_db:
            if is_fallen:
                self.track_db[track_id]['fall_counter'] += 1
            else:
                self.track_db[track_id]['fall_counter'] = 0
            return self.track_db[track_id]['fall_counter']
        return 0
