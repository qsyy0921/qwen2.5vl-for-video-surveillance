# vs_utils/motion.py
import cv2
import numpy as np

def detect_motion_roi(video_path: str, sample_rate=5, diff_thresh=30, area_thresh=500):
    """
    帧差运动检测，返回合并运动区域的 bbox
    """
    if video_path.startswith("file://"):
        video_path = video_path.replace("file://", "")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[motion] ❌ 无法打开视频: {video_path}")
        return None

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    accum_mask = None
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % sample_rate != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray

        _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        accum_mask = th if accum_mask is None else cv2.bitwise_or(accum_mask, th)

    cap.release()

    if accum_mask is None:
        return None

    contours, _ = cv2.findContours(accum_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_pts = []
    for c in contours:
        if cv2.contourArea(c) >= area_thresh:
            all_pts.extend(c.reshape(-1, 2).tolist())

    if not all_pts:
        return None

    pts = np.array(all_pts)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    bbox = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
    print(f"[motion] ✅ 检测到运动区域: {bbox}")
    return bbox
