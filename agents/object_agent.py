import requests
import cv2
import numpy as np
import tempfile
import os

API_URL = "http://127.0.0.1:8000/infer"

class ObjectAgent:
    """针对异常区域，分析对象与行为"""
    def __init__(self, session=None):
        self.s = session or requests.Session()

    def analyze(self, video_path: str, mask: dict = None):
        video_file = video_path
        if mask:
            cap = cv2.VideoCapture(video_path.replace("file://", ""))
            width = mask['x2'] - mask['x1']
            height = mask['y2'] - mask['y1']
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_file, fourcc, 1.0, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                roi = frame[mask['y1']:mask['y2'], mask['x1']:mask['x2']]
                roi = cv2.resize(roi, (width, height))
                out.write(roi)
            cap.release()
            out.release()
            video_file = f"file://{tmp_file}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_file, "fps": 1.0},
                {"type": "text", "text": "请分析异常区域内的关键对象及其行为与交互。"}
            ]
        }]
        r = self.s.post(API_URL, json=messages, timeout=300)
        r.raise_for_status()
        data = r.json()

        if mask:
            os.unlink(tmp_file)

        return data.get("result", data)
