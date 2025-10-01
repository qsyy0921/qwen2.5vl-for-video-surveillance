import cv2
from ultralytics import YOLO
import numpy as np
import logging
import contextlib
import os

# 禁用 Ultralytics 日志
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# YOLO 初始化，verbose=False
model = YOLO("/home/l40/newdisk1/mfl/videosur/models/yolov8n.pt", verbose=False)

# 屏蔽 stdout，防止 tqdm / print 输出
def suppress_output(func, *args, **kwargs):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        return func(*args, **kwargs)

def detect_anomaly_roi(video_path: str, conf_thresh=0.3):
    """
    使用 YOLO 检测视频异常/运动区域
    返回 ROI bbox {"x1":..,"y1":..,"x2":..,"y2":..} 或 None
    """
    if video_path.startswith("file://"):
        video_path = video_path.replace("file://", "")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    all_pts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 屏蔽 YOLO 内部输出
        results = suppress_output(model, frame)[0]

        for box in results.boxes:
            conf = float(box.conf)
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_pts.extend([[x1, y1], [x2, y2]])

    cap.release()
    if not all_pts:
        return None

    pts = np.array(all_pts)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
